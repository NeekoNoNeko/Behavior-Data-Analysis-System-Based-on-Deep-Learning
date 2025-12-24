from cleanvision import Imagelab
data_path = "C:\workspace\Behavior-Data-Analysis-System-Based-on-Deep-Learning\data\all_frames"
# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path=data_path)

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()

# Produce a neat report of the issues found in your dataset
imagelab.report()