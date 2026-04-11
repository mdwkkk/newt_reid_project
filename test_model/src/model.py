# ═══════════════════════════════════════════════════════════════
# model.py - Модели из segmentation_models.pytorch
# ═══════════════════════════════════════════════════════════════

import segmentation_models_pytorch as smp

# ═══════════════════════════════════════════════════════════════
# Настройки по умолчанию
# ═══════════════════════════════════════════════════════════════

DEFAULT_ARCH = 'unet'
DEFAULT_ENCODER = 'efficientnet-b3'
DEFAULT_PRETRAINED = True

# ═══════════════════════════════════════════════════════════════
# Создание модели
# ═══════════════════════════════════════════════════════════════

def create_model(
    arch=DEFAULT_ARCH,
    encoder=DEFAULT_ENCODER,
    pretrained=DEFAULT_PRETRAINED,
    in_channels=3,
    classes=1
):
    """
    Создаёт модель сегментации из segmentation_models.pytorch.
    
    Args:
        arch: Архитектура ('unet', 'unetplusplus', 'fpn', 'deeplabv3+', 'pspnet')
        encoder: Энкодер (см. доступные ниже)
        pretrained: Использовать предобученные веса ImageNet
        in_channels: Количество входных каналов (3 для RGB)
        classes: Количество классов (1 для бинарной сегментации)
    
    Returns:
        PyTorch модель
    """
    
    encoder_weights = 'imagenet' if pretrained else None
    
    if arch == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    
    elif arch == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    
    elif arch == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    
    elif arch == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    
    elif arch == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")
    
    return model

# ═══════════════════════════════════════════════════════════════
# Доступные энкодеры
# ═══════════════════════════════════════════════════════════════

ENCODERS = [
    # EfficientNet (рекомендую)
    'efficientnet-b0',  # Самый быстрый
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',  # Баланс скорость/точность
    'efficientnet-b4',  # Более точный
    'efficientnet-b5',  # Ещё точнее
    'efficientnet-b6',
    'efficientnet-b7',  # Самый точный (медленный)
    
    # ResNet
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    
    # MobileNet (для мобильных устройств)
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    
    # DenseNet
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    
    # VGG
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    
    # Inception
    'inceptionresnetv2',
    'inceptionv4',
    
    # Xception
    'xception',
    
    # MixVision Transformer
    'mit_b0',
    'mit_b1',
    'mit_b2',
    'mit_b3',
    'mit_b4',
    'mit_b5',
]

# ═══════════════════════════════════════════════════════════════
# Для обратной совместимости
# ═══════════════════════════════════════════════════════════════

class AttentionUNet:
    """Обёртка для совместимости со старым кодом"""
    def __new__(cls, n_channels=3, n_classes=1):
        return create_model(
            arch='unet',
            encoder='efficientnet-b3',
            pretrained=True,
            in_channels=n_channels,
            classes=n_classes
        )