from torchvision.datasets._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel
from torchvision.datasets._stereo_matching import (
    CarlaStereo,
    CREStereo,
    ETH3DStereo,
    FallingThingsStereo,
    InStereo2k,
    Kitti2012Stereo,
    Kitti2015Stereo,
    Middlebury2014Stereo,
    SceneFlowStereo,
    SintelStereo,
)
from .caltech import Caltech101, Caltech256
from torchvision.datasets.celeba import CelebA
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.clevr import CLEVRClassification
from torchvision.datasets.coco import CocoCaptions, CocoDetection
from .country211 import Country211
from .dtd import DTD
from .eurosat import EuroSAT
from torchvision.datasets.fakedata import FakeData
from torchvision.datasets.fer2013 import FER2013
from .fgvc_aircraft import FGVCAircraft
from torchvision.datasets.flickr import Flickr30k, Flickr8k
from .flowers102 import Flowers102
from torchvision.datasets.folder import DatasetFolder, ImageFolder
from .food101 import Food101
from torchvision.datasets.gtsrb import GTSRB
from torchvision.datasets.hmdb51 import HMDB51
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.imagenette import Imagenette
from torchvision.datasets.inaturalist import INaturalist
from torchvision.datasets.kinetics import Kinetics
from torchvision.datasets.kitti import Kitti
from torchvision.datasets.lfw import LFWPairs, LFWPeople
from torchvision.datasets.lsun import LSUN, LSUNClass
from torchvision.datasets.mnist import EMNIST, FashionMNIST, KMNIST, MNIST, QMNIST
from torchvision.datasets.moving_mnist import MovingMNIST
from torchvision.datasets.omniglot import Omniglot
from .oxford_iiit_pet import OxfordIIITPet
from .pcam import PCAM
from torchvision.datasets.phototour import PhotoTour
from torchvision.datasets.places365 import Places365
from torchvision.datasets.rendered_sst2 import RenderedSST2
from torchvision.datasets.sbd import SBDataset
from torchvision.datasets.sbu import SBU
from torchvision.datasets.semeion import SEMEION
from .stanford_cars import StanfordCars
from torchvision.datasets.stl10 import STL10
from .sun397 import SUN397
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.ucf101 import UCF101
from torchvision.datasets.usps import USPS
from .vision import VisionDataset
from torchvision.datasets.voc import VOCDetection, VOCSegmentation
from torchvision.datasets.widerface import WIDERFace
from .imagenet_100 import Imagenet100

__all__ = (
    "LSUN",
    "LSUNClass",
    "ImageFolder",
    "DatasetFolder",
    "FakeData",
    "CocoCaptions",
    "CocoDetection",
    "CIFAR10",
    "CIFAR100",
    "EMNIST",
    "FashionMNIST",
    "QMNIST",
    "MNIST",
    "KMNIST",
    "MovingMNIST",
    "StanfordCars",
    "STL10",
    "SUN397",
    "SVHN",
    "PhotoTour",
    "SEMEION",
    "Omniglot",
    "SBU",
    "Flickr8k",
    "Flickr30k",
    "Flowers102",
    "VOCSegmentation",
    "VOCDetection",
    "Cityscapes",
    "ImageNet",
    "Caltech101",
    "Caltech256",
    "CelebA",
    "WIDERFace",
    "SBDataset",
    "VisionDataset",
    "USPS",
    "Kinetics",
    "HMDB51",
    "UCF101",
    "Places365",
    "Kitti",
    "INaturalist",
    "LFWPeople",
    "LFWPairs",
    "KittiFlow",
    "Sintel",
    "FlyingChairs",
    "FlyingThings3D",
    "HD1K",
    "Food101",
    "DTD",
    "FER2013",
    "GTSRB",
    "CLEVRClassification",
    "OxfordIIITPet",
    "PCAM",
    "Country211",
    "FGVCAircraft",
    "EuroSAT",
    "RenderedSST2",
    "Kitti2012Stereo",
    "Kitti2015Stereo",
    "CarlaStereo",
    "Middlebury2014Stereo",
    "CREStereo",
    "FallingThingsStereo",
    "SceneFlowStereo",
    "SintelStereo",
    "InStereo2k",
    "ETH3DStereo",
    "wrap_dataset_for_transforms_v2",
    "Imagenette",
    "Imagenet100"
)


# We override current module's attributes to handle the import:
# from torchvision.datasets import wrap_dataset_for_transforms_v2
# without a cyclic error.
# Ref: https://peps.python.org/pep-0562/
def __getattr__(name):
    if name in ("wrap_dataset_for_transforms_v2",):
        from torchvision.tv_tensors._dataset_wrapper import wrap_dataset_for_transforms_v2

        return wrap_dataset_for_transforms_v2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")