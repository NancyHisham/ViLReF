
class feat_extract_img():
    def __init__(self, model):
        self.model = model

    def __call__(self, imgs):
        model = self.model
        if hasattr(model, "module"):
            return model.module.encode_image_featExt(imgs)
        else:
            return model.encode_image_featExt(imgs)
