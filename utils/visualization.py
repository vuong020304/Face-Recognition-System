import cv2
import matplotlib.pyplot as plt

def draw_faces(image, faces, results=None):
    """Vẽ khuôn mặt và kết quả nhận diện lên ảnh"""
    if not faces:
        return image.copy()
    
    img_draw = image.copy()
    
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        
        # Lấy thông tin kết quả
        result_info = results[i] if results and i < len(results) else None
        color = _get_color(result_info)
        
        # Vẽ bbox
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ text nếu có kết quả
        if result_info:
            label = f"{result_info['result']} ({result_info.get('score', 0):.2f})"
            cv2.putText(img_draw, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img_draw

def _get_color(result_info):
    """Xác định màu sắc dựa trên kết quả nhận diện"""
    if not result_info:
        return (0, 255, 0)  # Xanh lá mặc định
    
    result = result_info["result"]
    score = result_info.get("score", 0)
    
    if result == "Unknown":
        return (255, 0, 0)  # Đỏ
    elif score > 0.7:
        return (0, 255, 0)  # Xanh lá
    elif score > 0.5:
        return (255, 255, 0)  # Vàng
    else:
        return (255, 165, 0)  # Cam

def show_image(image, title=None):
    """Hiển thị ảnh bằng matplotlib"""
    plt.figure(figsize=(12, 8))
    
    # Hiển thị ảnh (giả định ảnh đã ở định dạng RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        plt.imshow(image)  # Ảnh RGB
    else:
        plt.imshow(image, cmap='gray')  # Ảnh xám
    
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

