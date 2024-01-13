##Hiệp ơi!!!

Lấy data về luật trong [file này nhé](/data/VLSP2023-LTER-Data/legal_passages.json).

Tạo một thư mục mới có tên là __generate_data__ rồi code trong forder này.

Lưu data mới vào file __arg_data_train.csv__ và __arg_data_test.csv__ rồi lưu vào thư mục __/data/datasets/arg/__ nhé.

Số lượng Sample trong tập test bằng 10% tổng số lượng trong tập train và tập test

Lưu í số lượng label 0 và 1 trong tập train phải tương đối cân đối. Không được chênh lệch quá nhiều.

File __arg_data.csv__ có dạng:
| text    | label    |
| :---: | :---: |
| Loại T18 (18+): Phim được phổ biến đến người xem từ đủ 18 tuổi trở lên   | 1 |
