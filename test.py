# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path

import numpy as np
import torch
from tqdm import tqdm

from utils import builder, configurator, io, misc, ops, pipeline, recorder
import time 
from thop import profile
import time 

def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/MRNet/cod_MRNet.py", type=str)
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs1_t.json", type=str)
    parser.add_argument("--model-name",default='MRNet',type=str)
    parser.add_argument("--batch-size",default=4, type=int)
    parser.add_argument("--load-from",default='D:/pythonCode/conpare/reTrain/xxx1/xxx/output/xxx_BS6_LR0.005_E4_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth', type=str)
    parser.add_argument("--save-path",default='D:/pythonCode/conpare/OrinDatasetTest/MRNet/re_our_2/result//', type=str)
    parser.add_argument("--minmax-results", action="store_true")
    parser.add_argument("--info", default='demo',type=str)
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)
    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size
    if args.load_from is not None:
        config.load_from = args.load_from
    if args.info is not None:
        config.experiment_tag = args.info
    if args.save_path is not None:
        if os.path.exists(args.save_path):
            if len(os.listdir(args.save_path)) != 0:
                print(os.listdir(args.save_path))
                # raise ValueError(f"--save-path is not an empty folder.")
        else:
            print(f"{args.save_path} does not exist, create it.")
            os.makedirs(args.save_path)
    config.save_path = args.save_path
    config.test.to_minmax = args.minmax_results

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            continue
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    return config


def test_once(
    model,
    data_loader,
    save_path,
    tta_setting,
    clip_range=None,
    show_bar=False,
    desc="[TE]",
    to_minmax=False,
):
    model.is_training = False
    cal_total_seg_metrics = recorder.CalTotalMetric()

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        # print(f'batch_images{batch["data"].shape}')
        batch_images = misc.to_device(batch["data"], device=model.device)
        for key, value in batch_images.items():
            print(f"Key: {key}, Shape: {value.shape}")
        if tta_setting.enable:
            logits = pipeline.test_aug(
                model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
            )
        else:
            logits = model(data=batch_images)
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

        for i, pred in enumerate(probs):
            mask_path = batch["info"]["mask_path"][i]
            mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape

            # here, sometimes, we can resize the prediciton to the shape of the mask's shape
            pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")

            if clip_range is not None:
                pred = ops.clip_to_normalize(pred, clip_range=clip_range)

            if to_minmax:
                pred = ops.minmax(pred)

            if save_path:  # 这里的save_path包含了数据集名字
                ops.save_array_as_image(data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path)

            pred = (pred * 255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


@torch.no_grad()
def testing(model, cfg):
    pred_save_path = None
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):
        if cfg.save_path:
            pred_save_path = os.path.join(cfg.save_path, data_name)
            print(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
        )
        print(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")


def main():
    
    cfg = parse_config()

    model, model_code = builder.build_obj_from_registry(
        registry_name="MODELS", obj_name=cfg.model_name, return_code=True
    )
    # io.load_weight(model=model, load_path=cfg.load_from)
    wd = torch.load(cfg.load_from)
    # print(wd.keys())
    model.load_state_dict(wd)
    model.device = "cuda:0"
    model.to(model.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.device = "cuda:0"
    model.to(device)

   # 创建一个示例输入张量并将其移动到相同的设备
    # input = torch.randn(1,4,384, 384).to(device)
#     batch_images = {
#     'image1.5': torch.randn(1, 3, 576, 576),
#     'image1.0': torch.randn(1, 3, 384, 384),
#     'image0.5': torch.randn(1, 3, 192, 192)
# }
#     for key, value in batch_images.items():
#         batch_images[key] = value.to(device)

#     # 计算FLOPs和参数量
#     flops, params = profile(net, inputs=(batch_images,))
#     # 将FLOPs转换为常用单位
#     flops_in_giga = flops / 10**9  # 转换为十亿（G）
#     flops_in_tera = flops / 10**12  # 转换为一万亿（T）

#     # 将参数量转换为常用单位
#     params_in_million = params / 10**6  # 转换为百万（M）
#     params_in_billion = params / 10**9  # 转换为十亿（B）

#     print(f"模型的FLOPs：{flops_in_giga:.2f} G or {flops_in_tera:.2f} T")
#     print(f"模型的参数量：{params_in_million:.2f} M or {params_in_billion:.2f} B")
#     # print(f"模型的FLOPs：{flops}")
#     # print(f"模型的参数量：{params}")

    # 估算单张图的运算时间
    # net.eval()
    # import time
    # with torch.no_grad():
    #     start_time = time.time()
    #     output = net(batch_images)
    #     end_time = time.time()

    # inference_time = end_time - start_time
    # print(f"单张图的运算时间：{inference_time} seconds")
    model.eval()
    t1  = time.time()
    testing(model=model, cfg=cfg)
    # print(f'test time')
    # print(time.time()-t1)

if __name__ == "__main__":
    main()
