import torch
import numpy as np
# from utils.boxs_utils import box_iou
from torchvision.ops import box_iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] #按置信度调整其顺序
    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):#按类别遍历
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases
            #在以-conf[i]为x,以recall[:,0]为y的范围内插入-pr_score【即构建了一个recall关于conf的函数】
            #批量得到在不同iou下的pr值

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):#【计算正常】
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

#注意：p和r主要计算iou>0.5的情况即可，不关注其他iou下的情况
def coco_map(predicts_list, targets_list):
    """
    :param predicts_list: per_img predicts_shape [n,6] (x1,y1,x2,y2,score,cls_id)
    :param targets_list: per_img targets_shape [m, 5] (cls_id,x1,y1,x2,y2)
    :return:
    """
    device = targets_list[0].device
    iouv = torch.linspace(0.5, 0.95, 10).to(device) # 生成不同的iou阈值
    niou = iouv.numel() #计算元素数目
    stats = list()
    for predicts, targets in zip(predicts_list, targets_list):
        nl = len(targets)
        tcls = targets[:, 0].tolist() if nl else []
        if predicts is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        correct = torch.zeros(predicts.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = list()
            tcls_tensor = targets[:, 0] #gt cls
            tbox = targets[:, 1:5] #gt box
            
            for cls in torch.unique(tcls_tensor):#按类别遍历
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1) #gt cls
                pi = (cls == predicts[:, 5]).nonzero(as_tuple=False).view(-1) #pred cls
                if pi.shape[0]:#【pred比gt多表现为框的冗余，造成查准率低查全率高】
                    ious, i = box_iou(predicts[pi, :4], tbox[ti]).max(dim=1) #计算pred与gt的iou
                    #值与索引
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):#在iou 50以上的结果才能参与后续运算
                        d = ti[i[j]]#以j为索引访问列表i
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv #计算某个iou在各种标准下的正确性
                        if len(detected) == nl:
                            break
        #TP pred_score pred_cls target_cls
        stats.append((correct.cpu(), predicts[:, 4].cpu(), predicts[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P@0.5, R@0.5, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean() #各类的均值
        return mp, mr, map50, map
    else:
        return 0., 0., 0., 0.


def coco_eval(anno_path="/home/huffman/data/annotations/instances_val2017.json", pred_path="predicts.json"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(anno_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
