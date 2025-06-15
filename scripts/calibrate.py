import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Opencv 相机自动标定软件",
        epilog="Example: python calibrtae.py -d left (校准左目相机内参)"
    )
    parser.add_argument(
        "-o","--order",
        default="left",
        help="calibrtae left or right cameras."
    )
    args = parser.parse_args()
    if args.order not in ["left","right"]:
        print("order must be left/right!")
        exit(-1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    chess_board_per_size = 0.02       # m

    pattern_size = (12,9)
    obj_points = np.zeros((pattern_size[0]*pattern_size[1],3), np.float16)
    obj_points[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    obj_points *= chess_board_per_size
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出...")
    corner_coordinates = []
    marker_corordinates = []
    full_corner_found = 0
     
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取帧")
            break
        if args.order == "left":
            frame = frame[:,:640]
        else:
            frame = frame[:, 640:]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
        ret, corners = cv2.findChessboardCorners(
            gray, 
            pattern_size, 
            None
        ) 
        if ret:
            refines_corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11), 
                (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(gray,pattern_size,refines_corners,ret)
         
        cv2.putText(gray, f"Find chees board, {full_corner_found}/15",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow(f"{args.order} original image",frame)
        cv2.imshow(f"{args.order} cheesboard match", gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('z'):
            if not ret:
                print("No chess board detection.")
            else:
                corner_coordinates.append(refines_corners)
                marker_corordinates.append(obj_points)
                full_corner_found += 1
        elif key == ord('x'): 
            if full_corner_found >= 15:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    marker_corordinates,
                    corner_coordinates,
                    gray.shape[::-1],
                    None,
                    None
                )  
                print("内参矩阵", mtx)
                print("畸变系数", dist)
            else:
                print("No enough image to calibrate.")
         
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
