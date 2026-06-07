from ultralytics import YOLO


if __name__ == '__main__':



    model = YOLO("./weights/best.pt")

    

    metrics = model.val(data="data.yaml")
    print(metrics.box.map)      # mAP50-95
    print(metrics.box.map50)    # mAP50
    print(metrics.box.mp)       # mean precision
    print(metrics.box.mr)       # mean recall