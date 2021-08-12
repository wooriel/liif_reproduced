import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.div2k import latent_coord

# import torch.autograd.profiler as profiler

class LIIF(nn.Module):
    def __init__(self, encoder, use_feature_unfold=True, use_local_ensemble=True, use_cell_decode=True):
        super().__init__()

        self.encoder = encoder
        self.feature = None
        self.h_coord = None
        self.use_fu = use_feature_unfold
        self.use_le = use_local_ensemble
        self.use_cd = use_cell_decode
        self.cont_coord = None


    def feature_unfolding(self):
        # convert hr_coord (latent vec) into 2D spectrum
        # hr_coord shape: (B, 64/32, h, w)
        # output shape: (B, 64/32*9, h, w)
        # nn.F.unfold() can be used instead <- actually this is faster since the loop is ran in cuda

        # padded_feature = F.pad(input=self.feature, pad=(1, 1, 1, 1), mode='constant', value=0)
        # sf_size = self.feature.size()
        # h = sf_size[2]
        # w = sf_size[3]
        # empty = []
        # for i in range(3):
        #     for j in range(3):
        #         empty.append(padded_feature[:, :, i:h+i, j:w+j])
        #         # print(padded_feature.size())
        # unfolded = torch.cat(empty, 1).contiguous()
        # self.feature = unfolded

        self.feature = F.unfold(self.feature, 3, padding=1).view(
                self.feature.shape[0], self.feature.shape[1] * 9, self.feature.shape[2], self.feature.shape[3])
                                                                                                                                                                                                                                                                                                                                                                           

    def cell_decode(self, cell, device):
        # with profiler.record_function("Cell Decode"):
        # cell = (B, C, h*w, 2)
        feat_size_t = self.feature.size()
        feat_size = torch.tensor([*feat_size_t[-2:]]).to(device)
        # multiply each cell size with width and height of the feature
        size_defined_cell = cell.mul(feat_size)
        # size_defined_cell2 = torch.stack((cell[:, :, 0].mul(feat_size_t[-2]),
        #         cell[:, :, 1].mul(feat_size_t[-1])), dim=2)
        return size_defined_cell


    def cont_represent(self, device):
        # with profiler.record_function("Local Ensemble"):
        c_feature = self.feature.clone()
        # c_feature_sz = c_feature.size()
        c_h_coord = self.h_coord.clone()
        # c_h_coord_sz = c_h_coord.size()
        self.h_coord = self.h_coord.unsqueeze(1) # (B, 1, H*W, 2)
        orig_coord = self.h_coord.permute(0, 3, 2, 1) # unshifted code (B, 2, H*W, 1)
        # orig_coord_sz = orig_coord.size()

        if self.use_le: # local ensemble
            orig_coord = orig_coord.expand(*orig_coord.size()[:-1], 4)\
                .clone().permute(0, 1, 3, 2) # (B, 2, H(1 or 4), W(2304))
            # orig_coord_sz = orig_coord.size()

            c_feat_height, c_feat_width = c_feature.size()[-2:]
            c_h, c_w = 1/c_feat_height, 1/c_feat_width
            tl = torch.tensor([-c_h+1e-6, -c_w+1e-6]).to(device)
            tr = torch.tensor([-c_h+1e-6, c_w-1e-6]).to(device)
            bl = torch.tensor([c_h-1e-6, -c_w+1e-6]).to(device)
            br = torch.tensor([c_h-1e-6, c_w-1e-6]).to(device)

            h_coord_tl = c_h_coord.add(tl) # tl: top_left
            h_coord_tr = c_h_coord.add(tr) # tr: top_right
            h_coord_bl = c_h_coord.add(bl) # bl: bottom_left
            h_coord_br = c_h_coord.add(br) # br: bottom_right

            # concatenate four corners of the h_coord
            h_corner_coord = torch.stack([h_coord_tl, h_coord_tr, h_coord_bl, h_coord_br], dim=1).to(device)
            h_corner_coord.clamp(-1+1e-6, 1-1e-6)
            self.h_coord = h_corner_coord
        
        # del c_h_coord
        # torch.cuda.empty_cache()
        c_h_coord = self.h_coord


        # calculate image_coordinate
        # c_feature (B, C, H, W)
        # c_h_coord (B, H(1 or 4), W(actually H*W), 2(x, y pair))
        c_feat_sz = c_feature.size()

        # dflag_img = torch.all(abs(c_feature)<=1)
        # grid in x, y order
        img_sample = F.grid_sample(input=c_feature, grid=c_h_coord.flip(-1), mode='nearest', padding_mode='reflection', align_corners=False)
        # isnan = not torch.all(torch.isnan(img_sample)) 
        # assert isnan
        # img_sample (B, C(576), H(1 or 4), W(2304))
        # img_sample_sz = img_sample.size()

        # lat_coord = latent_coord([c_feature.size()[0], *c_feature.size()[-2:]]).unsqueeze(1).to(device)
        # the latent_coord function output (B, H:48, W:48, 2) -> (B, 2, H, W)
        lat_coord = latent_coord([c_feat_sz[0], *c_feat_sz[-2:]], flatten=False)\
            .to(device).permute(0, 3, 1, 2)
        # isnan = not torch.all(torch.isnan(lat_coord))
        # assert isnan
        # lat_coord_sz = lat_coord.size()
        # map h_coord to generated x and y latent vector
        lat_sample = F.grid_sample(input=lat_coord, grid=c_h_coord.flip(-1), mode='nearest', padding_mode='reflection', align_corners=False)
        # isnan = not torch.all(torch.isnan(lat_sample))
        # assert isnan
        # lat_sample (B, 2, H(1 or 4), W(2304))
        # lat_sample_sz = lat_sample.size()

        # orig_coord (B, 2, H(1 or 4), W(2304))
        rel_sample = orig_coord.sub(lat_sample)
        # isnan = not torch.all(torch.isnan(rel_sample))
        # assert isnan
        # rel_sample (B, 2, H(1 or 4), W(2304))
        # rel_sample_sz = rel_sample.size()

        # coordinate in range [-1, 1] * [h, w]
        img_sz_tensor = torch.tensor([*c_feat_sz[-2:]]).to(device)
        # B C:2 H:4 W:2304 -> B H W C -> B C H W
        rel_sample = rel_sample.permute(0, 2, 3, 1).mul(img_sz_tensor).permute(0, 3, 1, 2)
        # rel_sample_sz = rel_sample.size()
        # img_sample
        cont_coord = torch.cat([img_sample, rel_sample], dim=1) # add on channel
        # cont_coord_sz (B, C(578), H(4), W(2304))
        # cont_coord_sz = cont_coord.size()


        # cont_coord = cont_coord.view(cont_coord.size()[0], -1, cont_coord.size()[3]) # B, 64*9, 4, 2304 -> B, 64*9*4, 2304

        area = None
        if self.use_le:
            rel_sample = rel_sample.permute(0, 3, 2, 1)
            areas = rel_sample[:, :, :, 0].mul(rel_sample[:, :, :, 1]).abs()
            # areas_sz = areas.size()

        return cont_coord, areas

    def encod_feat(self, lr_img):
        self.feature = self.encoder(lr_img)

    def help_forward(self, h_coord, cell, device):
        # self.feature = self.encoder(lr_img)
        # pad the image boundary with 0 - I won't pad the latent coordinate
        # self.feature = self.encoder(lr_img)
        self.h_coord = h_coord
        channel = self.feature.size()[1]
        if self.use_fu:
            self.feature_unfolding()
            channel *= 9
        # feature_size = self.feature.size()
        # h_coord_size = h_coord.size()


        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        self.cont_coord, areas = self.cont_represent(device)
        if self.use_le:
            # self.cont_coord, areas = self.local_ensemble(h_coord, cell, device)
            channel += 2 # coordinate +=2
            # channel *= 4 # calculate four dim concurrently
        # else:
            # self.cont_coord = self.feature.view(feature_size[:-2], -1)
            # areas = None
        if self.use_cd:
            rel_cell = self.cell_decode(cell, device)
            if self.use_le:
                # cont_coord_sz = self.cont_coord.size()
                rel_cell_sz = rel_cell.size()
                rel_cell = rel_cell.unsqueeze(-1).expand(*rel_cell_sz, 4).permute(0, 2, 3, 1).clone()
                # repeat(rel_cell_sz[:-2], 4, rel_cell_sz[-1])
                # rel_cell_sz = rel_cell.size()
            # cont_coord_sz (B, C(578), H(4), W(2304))
            # cont_coord_sz = self.cont_coord.size()
            self.cont_coord = torch.cat((self.cont_coord, rel_cell), dim=1)\
                    # .permute(0, 3, 2, 1).contiguous()
                    # .view(cont_coord_sz[0], cont_coord_sz[3], -1)
            channel += 2 # added rel_cell

        self.cont_coord = self.cont_coord.permute(0, 3, 2, 1).contiguous()

        # cont_coord_sz = self.cont_cosord.size()
        # cont_coord_sz (B, img_H*img_W(2304), 4, C(580))
        # but when entering the mlp layer, cont_coord should be in size (B*2304, 580*4)

        # always goes into MLP layer
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        return [self.cont_coord, areas, channel]

    def forward(self, lr_img, h_coord, cell, device):
        self.encod_feat(lr_img)
        return self.help_forward(h_coord, cell, device)