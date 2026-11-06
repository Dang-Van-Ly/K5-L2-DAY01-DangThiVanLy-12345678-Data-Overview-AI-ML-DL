# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:6/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=`ImageNet-1K`
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Record này nói về cả bức ảnh 
    Vì: 
    1. Trường "task" : "image_classification" cho biết mô hình đang thực hiện nhiệm vụ phân loại toàn bộ hình ảnh.
    2. Mỗi ảnh (sample_id) có nhiều nhãn dự đoán được xếp hạng từ 1 đến 5(rank)
    3. không có thông tin tọa độ để xác định vị trí từng vật thể
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp (class list) không phải do model tự nghĩ ra, mà được quy định sẵn và đi kèm với taxonomy/checkpoint của mô hình.
    
    1. Taxonomy: Là hệ thống phân loại chứa danh sách các lớp mà mô hình có thể nhận diện. Trong dữ liệu này là ImageNet-1K, gồm 1.000 lớp.
    2. Checkpoint: File trọng số mô hình (yolo11n-cls.pt) được huấn luyện để dự đoán các lớp tương ứng trong taxonomy.
    3. Model: Khi nhận ảnh đầu vào, model tính toán xác suất cho từng lớp có sẵn và trả về các lớp có điểm số cao nhất.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Vì mỗi trường có một vai trò khác nhau, giúp kết quả phân loại chính xác, dễ tra cứu, tránh nhầm lẫn.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Nếu trong ảnh có nhiều vật thể hoặc nhiều chủ đề, guideline cần quy định rõ tiêu chí chọn một lớp đại diện cho toàn bộ bức ảnh, tránh việc mỗi người gán nhãn chọn một lớp khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
    vì score chỉ thể hiện mức độ tự tin của mô hình với dự đoán, không đả bào dự đoán đó đúng, model có thể nhầm lần, dự đoán sai nhưng vẫn có score cao.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=``person; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Box bao quanh người trong ảnh kitchen:
    - Góc trên-trái: Nằm cách mép trái ảnh 385,33 pixel và cách mép trên ảnh 69,24 pixel.
    - Góc dưới-phải: Nằm cách mép trái ảnh 498,92 pixel và cách mép trên ảnh 348,92 pixel.
    - Chiều rộng: 113,58 pixel.
    - Chiều cao: 279,68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    Chọn hai ngưỡng: 0.20 và 0.60.
    Ngưỡng (threshold)	Số prediction
    0.20	            11
    0.60	            6
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Khi hạ ngưỡng confidence, AI sẽ giữ lại nhiều prediction (nhiều khung bao) hơn, bao gồm cả những dự đoán chưa chắc chắn hoặc sai. reviewer (người kiểm duyệt) phải làm thêm các công việc:
    1. Kiểm tra từng khung bao AI phát hiện.
    2. Xác nhận vật thể AI nhận diện có đúng không.
    3. Loại bỏ các khung bao sai hoặc trùng lặp.
    4. Chỉnh sửa vị trí, kích thước khung bao nếu chưa chính xác.
    5. Bổ sung những vật thể AI bỏ sót.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    “Người gán nhãn phải vẽ khung chữ nhật bao sát vật thể cần nhận diện, bao gồm toàn bộ phần vật thể nhìn thấy được, không lấy quá nhiều vùng nền xung quanh. Nếu vật thể bị che khuất, chỉ vẽ theo phần có thể quan sát được, không tự suy đoán phần bị che.”
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Một người ở gần mép ảnh bị cắt mất một phần cơ thể. Tôi sẽ đưa trường hợp này vào guideline nếu quy tắc xử lý đã rõ: chỉ vẽ bounding box bao quanh phần vật thể nhìn thấy được, không tự suy đoán phần bị cắt. Nếu chưa rõ có nên gán nhãn vật thể bị cắt một phần hay không, tôi sẽ hỏi Lab Coach để thống nhất cách xử lý.

    Vì guideline giúp người gán nhãn xử lý thống nhất những trường hợp phổ biến, còn escalation giúp giải quyết những tình huống chưa có quy định rõ ràng.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-009` ; `class_name`=`person` ; `score`=0.431437 ; số điểm polygon=`polygon_point_count`=65 ; 3–5 điểm đầu của `polygon_xy`=[
        1.0,
        264.0
      ],
      [
        1.0,
        309.0
      ],
      [
        5.0,
        309.0
      ],
      [
        6.0,
        308.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    Bounding box (box) chỉ tạo một hình chữ nhật bao quanh vật thể. Nó cho biết vị trí, chiều rộng và chiều cao, nhưng không mô tả chính xác đường viền của vật thể.

    Polygon sử dụng nhiều điểm tọa độ nối với nhau để mô tả đường viền thực tế của vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    instance_id dùng để định danh riêng từng đối tượng được phát hiện trong một ảnh.
    Hiểu câu "không phải class ID và không phải tracking ID"
    - Không phải class ID: Vì mỗi đối tượng có instance ID riêng, kể cả khi chúng cùng loại. Ví dụ hai người khác nhau vẫn có hai instance_id, nhưng cùng class_id = 0.
    - Không phải tracking ID: Vì instance_id chỉ định danh đối tượng trong dữ liệu ảnh hiện tại, không có nghĩa là theo dõi đối tượng đó xuyên suốt nhiều khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Người gán nhãn phải vẽ đường biên mask sát theo đường viền thực tế của vật thể, chỉ bao gồm phần nhìn thấy được, không lấy thêm vùng nền xung quanh. Nếu vật thể bị che khuất, chỉ vẽ phần có thể quan sát được, không tự suy đoán phần bị che.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Vấn đề: Người nằm sát mép ảnh, một phần đường viền có thể bị giới hạn bởi cạnh ảnh.
    - Cách xử lý: Dùng guideline nếu đã có quy định về vật thể bị cắt bởi mép ảnh.
    - Nếu guideline chưa quy định rõ: Escalation (báo Lab Coach hoặc người phụ trách) để xin hướng dẫn trước khi tự quyết định.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ        | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì?|
|---|---|---|---|---|
| Phân loại ảnh | Một hoặc nhiều nhãn đại diện cho toàn bộ ảnh, ví dụ: kitchen, traffic, dining. | Khó chọn nhãn đại diện nếu ảnh có nhiều loại đối tượng. | Gán đúng nhãn cho toàn bộ ảnh theo guideline. | Kiểm tra nhãn có đúng nội dung tổng thể của ảnh không. |
| Phát hiện vật thể | Mỗi vật thể có class_id, class_name và một bounding box [x1,y1,x2,y2]. | JSON kitchen có 11 prediction. Cần kiểm tra box có sát vật thể không, có bỏ sót hoặc nhận diện nhầm không. | Gán nhãn từng vật thể, chọn đúng class và vẽ box bao quanh vật thể. | Kiểm tra class, box, vật thể bị bỏ sót và box dư. |
| Instance segmentation | Mỗi vật thể có instance_id, class_name và polygon/mask mô tả đường viền. | JSON kitchen-009 có polygon_point_count = 65, score = 0.431437; đối tượng nằm sát mép trái ảnh. | Vẽ mask sát đường viền phần vật thể nhìn thấy. | Kiểm tra mask có sát biên không, có lấn sang nền hoặc thiếu phần vật thể không. |
## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Không chia sẻ, phát tán hoặc sử dụng dữ liệu của bài lab ra ngoài phạm vi được cho phép; không tự ý chỉnh sửa hoặc tiết lộ thông tin nhạy cảm.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Tôi sẽ dừng xử lý dữ liệu đó, không tải xuống hoặc chia sẻ thêm, sau đó báo cho Lab Coach hoặc người phụ trách lab để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
