def pos_get(video):

    import cv2
    from pose_cls_lib import create_inferencer

    cap = cv2.VideoCapture(video)  # 替换为你的视频路径
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率[2,4](@ref)
    frame_interval = int(2*fps)  # 计算2秒对应的帧数


    frame_count = 0 # 初始化帧计数器
    second_count = 0 # 初始化秒计数器
    sit_pos = ['其他'] # 坐姿记录列表
    sit_time = [] # 坐姿持续时间列表

    while True:
        # 设置当前读取位置[4,5](@ref)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Frame', frame)

        #处理图像
        infer = create_inferencer(
            pose_onnx="D:/programm/Jupyter/DL/AI+m/onnx/model/yolo11n-pose.onnx",
            cls_onnx="D:/programm/Jupyter/DL/AI+m/onnx/model/skeleton-cls.onnx",
            class_names=["其他", "正坐", "托着下巴", "驼背"]
        )

        label = infer(frame)
        #坐姿持续时间
        if label != sit_pos[-1]:
            sit_time.append(sit_pos[-1] + ':' + str(second_count) + '秒')
            sit_pos[-1] = label
            second_count = 0
            print('切换到' + label)
        else:
            second_count += 2
            print('持续' + label + ':' + str(second_count) + '秒')



        #计时更新
        frame_count += frame_interval  # 更新下一帧位置
        second_count += 2 # 更新下一秒位置

        # 等待约30毫秒并检测q键，保持界面响应[1](@ref)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(sit_time)
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(str(sit_time))


    return sit_time

pos_get('test.mp4')
