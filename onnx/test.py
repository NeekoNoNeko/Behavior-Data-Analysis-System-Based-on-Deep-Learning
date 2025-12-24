from pose_cls_lib import create_inferencer

infer = create_inferencer(
    pose_onnx="pose.onnx",
    cls_onnx="cls.onnx",
    class_names=["other", "sit", "resting_chin", "hunchback"]
)

label = infer("test/2.jpg")
print(label)
