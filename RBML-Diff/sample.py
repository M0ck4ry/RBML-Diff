import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.train_utils import set_random_seed
from utils import init_env
import os
import argparse
from pathlib import Path
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

# 设置随机种子为 7，用于保证实验的可重复性
set_random_seed(7)

# 添加计算模型参数量和大小的函数
def get_model_size(model):
    """计算模型参数量和大小"""
    total_params = sum(p.numel() for p in model.parameters())
    # 假设所有参数都是float32类型，每个参数占4字节
    total_size = total_params * 4 / (1024 * 1024)  # 转换为MB
    return total_params, total_size

def get_loader(cfg):
    Kvasir_test_dataset = instantiate_from_config(cfg.test_dataset.Kvasir)
    CVC_300_test_dataset = instantiate_from_config(cfg.test_dataset.CVC_300)
    CVC_ClinicDB_test_dataset = instantiate_from_config(cfg.test_dataset.CVC_ClinicDB)
    CVC_ColonDB_test_dataset = instantiate_from_config(cfg.test_dataset.CVC_ColonDB)
    ETIS_LaribPolypDB_test_dataset = instantiate_from_config(cfg.test_dataset.ETIS_LaribPolypDB)
    Kvasir_test_loader = DataLoader(
        Kvasir_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    CVC_300_test_loader = DataLoader(
        CVC_300_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    CVC_ClinicDB_test_loader = DataLoader(
        CVC_ClinicDB_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    CVC_ColonDB_test_loader = DataLoader(
        CVC_ColonDB_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    ETIS_LaribPolypDB_test_loader = DataLoader(
        ETIS_LaribPolypDB_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return  Kvasir_test_loader, CVC_300_test_loader, CVC_ClinicDB_test_loader, CVC_ColonDB_test_loader, ETIS_LaribPolypDB_test_loader

def plot_inference_speed(dataset_names, fps_values, output_path):
    """绘制推理速度柱状图并保存"""
    plt.figure(figsize=(12, 6))
    
    # 创建柱状图
    bars = plt.bar(dataset_names, fps_values, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f} FPS',
                 ha='center', va='bottom', fontsize=10)
    
    # 设置图表标题和标签
    plt.title('各数据集推理速度比较', fontsize=14)
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('推理速度 (FPS)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"推理速度图表已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../...model-149.pt')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='../results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    parser.add_argument('--target_dataset', nargs='+', type=str, default=['Kvasir', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB'])
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    Kvasir_test_loader, CVC_300_test_loader, CVC_ClinicDB_test_loader, CVC_ColonDB_test_loader, ETIS_LaribPolypDB_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)
    # 优化器
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    Kvasir_test_loader, CVC_300_test_loader, CVC_ClinicDB_test_loader, CVC_ColonDB_test_loader, ETIS_LaribPolypDB_test_loader  = \
        trainer.accelerator.prepare(Kvasir_test_loader, CVC_300_test_loader, CVC_ClinicDB_test_loader, CVC_ColonDB_test_loader, ETIS_LaribPolypDB_test_loader)
    
    # 计算并输出模型参数量和大小
    if trainer.accelerator.is_main_process:
        total_params, total_size = get_model_size(trainer.model)
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"模型大小: {total_size:.2f} MB")

    dataset_map = {
        'Kvasir': Kvasir_test_loader,
        'CVC-300': CVC_300_test_loader,
        'CVC-ClinicDB': CVC_ClinicDB_test_loader,
        'CVC-ColonDB': CVC_ColonDB_test_loader,
        'ETIS-LaribPolypDB' : ETIS_LaribPolypDB_test_loader
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset] # dataset_loader对象的一个大列表5个

    # 记录总推理时间和总样本数
    total_inference_time = 0
    total_samples = 0
    
    # 存储每个数据集的FPS
    dataset_fps = []
    dataset_names = []
    
    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        mask_path = Path(cfg.test_dataset.Kvasir.params.image_root).parent.parent   
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        
        # 记录当前数据集的推理时间
        start_time = time.time()
        
        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            # 运行，这里会跳入
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        
        # 计算当前数据集的推理时间
        dataset_inference_time = time.time() - start_time
        dataset_sample_count = len(dataset.dataset) if hasattr(dataset.dataset, '__len__') else sum(1 for _ in dataset)
        
        # 更新总时间和总样本数
        total_inference_time += dataset_inference_time
        total_samples += dataset_sample_count
        
        # 计算当前数据集的平均推理时间
        avg_time_per_sample = dataset_inference_time / dataset_sample_count if dataset_sample_count > 0 else 0
        fps = dataset_sample_count / dataset_inference_time if dataset_inference_time > 0 else 0
        
        # 存储FPS值用于绘图
        dataset_fps.append(fps)
        dataset_names.append(dataset_name)
        
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')
        if trainer.accelerator.is_main_process:
            print(f"{dataset_name} 推理时间: {dataset_inference_time:.2f}秒")
            print(f"{dataset_name} 样本数量: {dataset_sample_count}")
            print(f"{dataset_name} 平均每张图片推理时间: {avg_time_per_sample*1000:.2f}毫秒")
            print(f"{dataset_name} FPS: {fps:.2f}帧/秒")

        if trainer.accelerator.is_main_process:
            from utils.eval import eval

            eval_score = eval(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
    
    # 输出总体推理性能统计
    if trainer.accelerator.is_main_process and total_samples > 0:
        overall_avg_time = total_inference_time / total_samples
        overall_fps = total_samples / total_inference_time
        print(f"\n总体推理性能统计:")
        print(f"总推理时间: {total_inference_time:.2f}秒")
        print(f"总样本数量: {total_samples}")
        print(f"总体平均每张图片推理时间: {overall_avg_time*1000:.2f}毫秒")
        print(f"总体FPS: {overall_fps:.2f}帧/秒")
        
        # 生成并保存推理速度图表
        output_path = Path(cfg.results_folder) / "inference_speed_comparison.png"
        plot_inference_speed(dataset_names, dataset_fps, output_path)