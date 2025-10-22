#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import datetime  # 用于生成带时间戳的文件名

def main_loop():
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光，曝光时间20ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 20 * 1000)
    
    # 白平衡
    mvsdk.CameraSetOnceWB(hCamera)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算buffer大小
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # 分配RGB buffer
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    # ==================== 新增：视频录制相关变量 ====================
    is_recording = False                      # 是否正在录制
    video_writer = None                       # OpenCV视频写入对象
    output_filename = ""                      # 输出视频文件名
    # ================================================================

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        # 从相机取一帧图片
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            
            # 转换为numpy数组
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Press q to end, s to start recording, e to stop", frame)

            # ==================== 新增：按键检测与视频录制控制 ====================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 按下 's' 开始录制
                if not is_recording:
                    # 生成带时间戳的文件名，避免重复
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"recording_{timestamp}.avi"
                    
                    # 获取当前帧的宽高和通道数
                    height, width = frame.shape[:2]
                    fps = 25  # 可以调整为你想要的帧率，比如 25, 30

                    # 选择编码器，通常用 MJPG (Motion-JPEG) 格式，兼容性好
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 或者使用 'XVID' 如果需要 MP4

                    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                    
                    if video_writer.isOpened():
                        is_recording = True
                        print(f"✅ 开始录制视频，保存为: {output_filename}")
                    else:
                        print("❌ 无法创建视频文件，请检查路径或编码器！")

            elif key == ord('e'):  # 按下 'e' 停止录制
                if is_recording:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    is_recording = False
                    print("⏹️ 停止录制视频")

            elif is_recording:  # 如果正在录制，且不是切换按键，则写入当前帧
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(frame)

            # ===================================================================

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    # ==================== 程序退出前释放资源 ====================
    # 如果还在录制，先停止录制
    if is_recording:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        is_recording = False
        print("🛑 程序退出前已停止录制")

    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)

def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()

main()
