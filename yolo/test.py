from ultralytics import YOLO

model = YOLO(r"C:\workspace\Behavior-Data-Analysis-System-Based-on-Deep-Learning\yolo\runs\cls\gesturev2\weights\best.pt")

results = model(r"test/3.jpg")

print(results[0].probs.top1)
print(results[0].names)
