from models.actors.base_actor import BaseActor
from utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from utils.heapmap_utils import generate_heatmap
from utils.ce_utils import generate_mask_cond, adjust_keep_rate


class SOVTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        out_dict = self.forward_pass(data)  # Backbone 输出 dict
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_image_list = []
        template_event_list = []
        template_flow_list = []
        search_image_list = []
        search_event_list = []
        search_flow_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_image_list.append(template_img_i)

            template_evt_i = data['template_events'][i].view(-1, *data['template_events'].shape[2:])
            template_event_list.append(template_evt_i)

            template_flo_i = data['template_flows'][i].view(-1, *data['template_flows'].shape[2:])
            template_flow_list.append(template_flo_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_image_list.append(search_img_i)

            search_evt_i = data['search_events'][i].view(-1, *data['search_events'].shape[2:])
            search_event_list.append(search_evt_i)

            search_flo_i = data['search_flows'][i].view(-1, *data['search_flows'].shape[2:])
            search_flow_list.append(search_flo_i)

        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_image_list[i].shape[0],
                                                     template_image_list[i].device,
                                                     data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # if len(template_list) == 1:
        #     template_list = template_list[0]

        out_dict = self.net(template_image=template_image_list,
                            template_event=template_event_list,
                            search_image=search_image_list,
                            search_event=search_event_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  # 定义 0 tensor，并指定GPU设备

        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                                 self.cfg.MODEL.BACKBONE.STRIDE)

        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss

            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss

            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss

            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss

            if return_status:
                # status for log
                status = {}

                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_IoU": mean_iou.item()}

                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
