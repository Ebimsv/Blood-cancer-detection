import torch

config = {
    'img_width': 224,
    'img_height': 224,
    'mean': [0.4527],
    'std': [0.1973],
    'name': 'resnet_50',
    'train_dir': 'data2/train',
    'test_dir': 'data2/test',
    'root_dir': 'data/C-NMC_test_prelim_phase_data',
    'csv_path': 'data/C-NMC_test_prelim_phase_data_labels.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'leaky_relu': False,
    'epochs': 100,
    'batch_size': 100,
    'eval_batch_size': 256,
    'seed': 42,
    'lr': 0.0001,
    'save_interval': 1,
    'reload_checkpoint': None,
    'finetune': 'weights/FA_DOCS/crnn-fa-base.pt',
    # 'finetune': None,
    'weights_dir': 'weights',
    'log_dir': 'logs',
    'cpu_workers': 4,
}

