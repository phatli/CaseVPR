from casevpr.frontend import input_transform, input_transform_boq, input_transform_vgt
from casevpr.utils import ARGS
from casevpr.models import NetVLAD, TVGNet, GeMPooling, HVPR_CaseNet, HVPR_SeqNet, JistModel, SVPR, imgVPR, imgDonothing, Crica, VPRModel

img_encoders = {
    "none": {
        "class": imgDonothing,
        "class_args": {},
        "img_shape": (224, 224),
        "ckpt_path": None,
        "input_transform": input_transform
    },
    "netvlad_WPCA4096": {
        "class": imgVPR,
        "class_args": {
            "img_model_name": "netvlad_WPCA4096",
            "arch": "vgg16",
            "pooling": "netvlad",
            "pool_class": NetVLAD,
            "pool_class_args": {
                "num_clusters": 64,
                "vladv2": False
            },
        },
        "ckpt_path": "models/netvlad_WPCA4096.pth.tar",
        "img_shape": (480, 640),
        "input_transform": input_transform_vgt

    },
    "ep": {
        "class": imgVPR,
        "class_args": {
            "img_model_name": "r18gemrbc",
            "arch": "resnet18",
            "pooling": "gem",
            "pool_class": GeMPooling,
            "pool_class_args": {},
        },
        "ckpt_path": "models/r18gemrbc.pth",
        "img_shape": (384, 384),
        "input_transform": input_transform_vgt
    },
    "crica": {
        "class": imgVPR,
        "class_args": {
            "img_model_name": "crica",
            "arch": "dinov2",
            "pooling": "crica",
            "pool_class": Crica,
            "pool_class_args": {},
        },
        "ckpt_path": "models/crica.pth",
        "img_shape": (224, 224),
        "input_transform": input_transform_vgt
    },
    "salad": {
        "hub": {
            "repo": "serizba/salad",
            "model": "dinov2_salad",
            "kwargs": {}
        },
        "ckpt_path": "models/dino_salad.ckpt",
        "img_shape": (322, 322),
        "input_transform": input_transform
    },
    "cliquemining": {
        "hub": {
            "repo": "serizba/salad",
            "model": "dinov2_salad",
            "kwargs": {}
        },
        "ckpt_path": "models/cliquemining.ckpt",
        "img_shape": (322, 322),
        "input_transform": input_transform
    },
    "boq": {
        "hub": {
            "repo": "amaralibey/bag-of-queries",
            "model": "get_trained_boq",
            "kwargs": {
                "backbone_name": "dinov2",
                "output_dim": 12288
            },
        },
        "ckpt_path": None,
        "img_shape": (322, 322),
        "input_transform": input_transform_boq
    },
    "mixvpr": {
        "class": VPRModel,
        "class_args": {
            "backbone_arch": "resnet50",
            "layers_to_crop": [4],
            "agg_arch": "MixVPR",
            "agg_config": {
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4
            }
        },
        "ckpt_path": "models/r50_mixvpr.ckpt",
        "img_shape": (320, 320),
        "input_transform": input_transform
    }
}

seq_encoders = {
    "vgg16_seqvlad": {
        "class": TVGNet,
        "class_default_args": ARGS,
        "class_args": {
            "arch": "vgg16",
            "aggregation": "seqvlad",
            "seq_length": 5,
        },
        "ckpt_path": "models/vgg16_seqvlad.pth",
        "seq_pca_path": "models/vgg16_seqvlad_pca.pkl",
        "img_shape": (384, 384),
        "input_transform": input_transform_vgt
    },
    "jist": {
        "class": JistModel,
        "class_args": {
            "backbone": "ResNet18",
            "fc_output_dim": 512,
            "agg_type": "seqgem",
            "seq_length": 5,
        },
        "ckpt_path": "models/r18_jist_pretrained.pth",
        "img_shape": (384, 384),
        "input_transform": input_transform_vgt
    },
    "svpr": {
        "class": SVPR,
        "class_args": {
            "arch": "stformer",
            "part": None,
            "seq_length": 5,
            "trunc_te": 4,
            "trunc_te_tatt": 4,
            "freeze_te": -1,
            "freeze_te_tatt": -1,
            "rel_pos_temporal": True,
            "rel_pos_spatial": True,
            "clusters": 64,
        },
        "ckpt_path": "models/svpr_pretrained.pth",
        "img_shape": (384, 384),
        "input_transform": input_transform_vgt
    }
}

hvpr_encoders = {
    "hvpr_casevpr_224_crica":{
        "class": HVPR_CaseNet,
        "class_args": {
            "seq_len": 5,
            "encoder_type": "bs_d_c"
        },
        "ckpt_path": "models/crica.pth",
        "img_shape": (224, 224),
        "input_transform": input_transform,
    },
    "hvpr_casevpr_224":{
        "class": HVPR_CaseNet,
        "class_args": {
            "seq_len": 5,
            "encoder_type": "bs_d_c"
        },
        "ckpt_path": "models/casevpr_224.pth",
        "img_shape": (224, 224),
        "input_transform": input_transform_vgt,
    },
    "hvpr_casevpr_322":{
        "class": HVPR_CaseNet,
        "class_args": {
            "seq_len": 5,
            "encoder_type": "bs_d_c"
        },
        "ckpt_path": "models/casevpr_322.pth",
        "img_shape": (322, 322),
        "input_transform": input_transform,
    },
    "hvpr_seqnet": {
        "class": HVPR_SeqNet,
        "class_args": {
            "w": 3,
            "arch": "vgg16",
            "output_dim": 4096,
            "seq_len": 5
        },
        "ckpt_path": "models/netvladwpca4096_seqnet.pth",
        "img_shape": (480, 640),
        "input_transform": input_transform_vgt,
    }
}
