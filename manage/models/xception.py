import timm

NAME = "xception"

def create_model(input_shape, config):
    in_chans = input_shape[0]
    num_classes = config["num_classes"]

    model = timm.create_model(
        "xception71",
        pretrained=True,
        in_chans=in_chans,
        num_classes=num_classes,
    )
    return model