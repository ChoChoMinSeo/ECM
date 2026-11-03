import torch.utils
from src.datasets.datasets import *
import torch
from torchvision import transforms

def get_train_dataset(args, subsample = 1, max_len = 100):
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(args.patch_size,pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    datasets = []
    if 'Kodak' in args.tasks:
        if args.wk == 'local':
            path = [
                "./data/CLIC2020/train/mobile_train_2020/train/","./data/CLIC2020/train/professional_train_2020/train/",\
            ]
        datasets.append(HQDataset(path, train_transforms, 'train',subsample, (max_len*args.natu)//2))
        datasets[-1].cls_idx = onehot_convert([0], max_cls = args.n_classes, eta = args.eta)
        if args.wk == 'local':
            path = [
                "./data/Flicker2W/train.list"
            ]
        datasets.append(HQDataset(path, train_transforms, 'train',subsample, (max_len*args.natu)//2))
        datasets[-1].cls_idx = onehot_convert([0], max_cls = args.n_classes, eta = args.eta)
        
    if 'Document' in args.tasks:
        if args.wk == 'local':
            path = "./data/2024NeurIPS/"
        datasets.append(DocumentDataset(path, train_transforms, 'train',subsample, max_len*args.docu))        
        datasets[-1].cls_idx = onehot_convert([1], max_cls = args.n_classes, eta = args.eta)
        
    if 'Residue' in args.tasks:
        if args.wk == 'local':
            path = "./data/vimeo_triplet/"
        datasets.append(VimeoTripletResidueDataset(path, train_transforms, 'train',subsample, max_len*args.diff))     
        datasets[-1].cls_idx = onehot_convert([2], max_cls = args.n_classes, eta = args.eta)
        
    if 'Y1Y2Y3' in args.tasks:
        if args.wk == 'local':
            path = "./data/vimeo_triplet/"
        datasets.append(VimeoTripletY1Y2Y3Dataset(path, train_transforms, 'train',subsample, max_len*args.moti)) 
        datasets[-1].cls_idx = onehot_convert([3], max_cls = args.n_classes, eta = args.eta)
        
    if 'KITTI' in args.tasks:
        if args.wk == 'local':
            path = "./data/Kitti/Kitti/"
        datasets.append(KittiDrivingDataset(path, train_transforms, 'train',subsample, max_len*args.kitt))
        datasets[-1].cls_idx = onehot_convert([4], max_cls = args.n_classes, eta = args.eta)
        
    if 'LineDrawing' in args.tasks:
        if args.wk == 'local':
            path = "./data/Manga109/Manga109_released_2023_12_07/"
        datasets.append(LineDataset(path, train_transforms, 'train',subsample, max_len*args.line))
        datasets[-1].cls_idx = onehot_convert([5], max_cls = args.n_classes, eta = args.eta)
        
        
    return torch.utils.data.ConcatDataset(datasets)

def get_test_dataset(args):
    test_transforms2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    datasets = []
    if 'Kodak' in args.test_tasks:
        if args.wk =='local':
            path = "./data/test/00_Kodak/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([0],max_cls=args.n_classes, eta = args.eta)
    if 'Document' in args.test_tasks:
        if args.wk == 'local':
            path = "./data/test/01_Document/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))
        datasets[-1].cls_idx= onehot_convert([1],max_cls=args.n_classes, eta = args.eta)
    if 'Residue' in args.test_tasks:
        if args.wk == 'local':
            path = "./data/test/02_Residue/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([2],max_cls=args.n_classes, eta = args.eta)
    if 'Y1Y2Y3' in args.test_tasks:
        if args.wk == 'local':
            path = "./data/test/03_Y1Y2Y3/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([3],max_cls=args.n_classes, eta = args.eta)
    if 'KITTI' in args.test_tasks:
        if args.wk == 'local':
            path = "./data/test/04_KITTI/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([4],max_cls=args.n_classes, eta = args.eta)
    if 'LineDrawing' in args.test_tasks:
        if args.wk == 'local':
            path = "./data/test/05_LineDrawing/"
        datasets.append(HQDataset(path, test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([5],max_cls=args.n_classes, eta = args.eta)
    
    return torch.utils.data.ConcatDataset(datasets)

def get_test_dataset2(args):
    datasets = []
    test_transforms2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if args.wk == 'local':
        path = "./data/test/00_CLIC/"
    datasets.append(HQDataset(path, test_transforms2, 'test'))
    datasets[-1].cls_idx= onehot_convert([0], max_cls = args.n_classes, eta = args.eta)
    return torch.utils.data.ConcatDataset(datasets)

dataset_name = ['Kodak','Document','Residue','Y1Y2Y3', 'KITTI','LineDrawing']
name_to_token_idx = {
    'Kodak':0,
    'Document':1,
    'Residue':2,
    'Y1Y2Y3': 3,
    'KITTI':4,
    'LineDrawing':5,
}
token_idx_to_name = {
    0:'Kodak',
    1:'Document',
    2:'Residue',
    3:'Y1Y2Y3',
    4:'KITTI',
    5:'LineDrawing',
}
task_colors = {
    'Kodak': 'blue',
    'natural_clic': 'red',
    'Residue': 'yellow',       
    'Y1Y2Y3': '#808080',
    'Document': 'green',           
    'KITTI': 'cyan',       
    'LineDrawing':'orange',
}

# Baseline
initial_token_stats = {
    'Kodak': { 
        'rate':[0.130501825983326, 0.198247019511958, 0.298263602269192, 0.436732123295466, 0.613576415305336, 0.839287536839644],
        "psnr":[29.639518737793, 31.1419787406921, 32.6875612735748, 34.4089248975118, 36.1523416837056, 37.8984320958455]
    },
    'Y1Y2Y3': {
        'rate':[0.118075927866525, 0.174807683025536, 0.250423573153584, 0.356458780551819, 0.491168009803483, 0.657239772771534],
        "psnr":[30.2256191906176, 31.623153887297, 32.9919894619992, 34.3618637386121, 35.6529975188406, 36.8295918765821]
    },
    'Residue': {
        'rate':[0.100120745697304, 0.150023099623228, 0.212542727025912, 0.298987055883596, 0.408005869898357, 0.544237288940502],
        "psnr":[31.9686923779939, 33.402616049114, 34.8648554400394, 36.4047510749415, 37.8728547849153, 39.3316957072208]
    },
    'Document': {
        'rate':[0.127059356775135, 0.172528396360576, 0.198893185394506, 0.256420342251658, 0.325576970974604, 0.412838665768504],
        "psnr":[32.7577881813049, 34.9365599950155, 37.0336861610413, 39.1689858436584, 41.2791113853455, 43.2500732739767]
    },
    'KITTI': {
        'rate':[0.129022969227088, 0.197425905222956, 0.293146602025157, 0.430773776612784, 0.614889005297109, 0.847076833248138],
        "psnr":[28.0591971748754, 29.1222340935155, 30.1910824524729, 31.2466706225747, 32.2122966866744, 33.0488339474327]
    },
    'LineDrawing': {
        'rate':[0.25854535587132, 0.355178575380705, 0.481705743237399, 0.663065082626417, 0.878920761519112,1.13822427997366],
        "psnr":[28.3602757453918, 30.1024646162987, 31.9450402259827, 33.9956914782524, 36.2073285579681, 38.6068336963654]
    },
    'natural_clic': { 
        'rate':[0.088690736353761, 0.131515826121336, 0.189950830507569, 0.271658880681526, 0.379913284102591, 0.5270599167],
        "psnr":[32.1219664783013, 33.5282310392798, 34.9188583187941, 36.3877449966058, 37.8466632191728, 39.33175872]
    },
}

trained_token_stats = {
    'Kodak': {
        'rate': [0.1347829305256406, 0.2076572123914957, 0.3063980316122373, 0.4467958528548479, 0.6276251946886381, 0.8528024852275848], 
        'psnr': [29.576042652130127, 31.05849488576253, 32.63069494565328, 34.35493977864584, 36.11925967534384, 37.90800555547079]
    },
    'Y1Y2Y3': {
        'rate': [0.109180397035456, 0.15742648226258, 0.221641813600926, 0.3157976536864513, 0.4460876464451614, 0.6153589527269727], 
        'psnr': [30.98524163898669, 32.5931541041324, 33.9872001095822, 35.43891901718943, 36.78027913444921, 38.11318794049715]
    },
    'Residue': {
        'rate': [0.0972460563245572, 0.1445943796762118, 0.2072976339902532, 0.2956928188275349, 0.4035481945856621, 0.5370507150594341], 
        'psnr': [32.092615729884095, 33.620137189563955, 35.10119413074694, 36.67788967333342, 38.22764050333124, 39.77225800564415]
    },
    'Document': {
        'rate': [0.0895725873609383, 0.1089336047880351, 0.1335775318245093, 0.160765572451055, 0.2061891726528605, 0.2343893510599931], 
        'psnr': [35.05819749832153, 37.15442657470703, 38.94783639907837, 40.80094941457113, 42.50830109914144, 44.21953137715658]
    },
    'KITTI': {
        'rate': [0.1427432635897084, 0.2321843465692118, 0.3258248520524878, 0.4771066441347724, 0.6763356673090082, 0.9156890244860398], 
        'psnr': [28.4846339476736, 29.77351916463752, 31.12714732320685, 32.56830561788458, 33.90704782385575, 35.19485242743241]
    },
    'LineDrawing': {
        'rate': [0.2403401704505086, 0.3217332869535312, 0.4381676883203909, 0.6116897322935984, 0.8320195304695517, 1.0683725327253342], 
        'psnr': [28.80265897512436, 30.497988879680637, 32.24685877561569, 34.310214042663574, 36.607354044914246, 39.03572845458984]
    },
    'natural_clic': {
        'rate': [0.0889804507719307, 0.1328137718686243, 0.1908597552739992, 0.2733554378515336, 0.3838795788767861, 0.5317263628651456], 
        'psnr': [32.10265196823492, 33.51174963974371, 34.90964587141828, 36.377955180842704, 37.859488510504, 39.36052927156774]
    },
}
