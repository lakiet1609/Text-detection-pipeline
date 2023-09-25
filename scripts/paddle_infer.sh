# python3 tools/infer/predict_det.py --det_algorithm="DB" --det_model_dir="en_PP-OCRv3_det_infer" --image_dir="./doc/imgs/" --use_gpu=True
# python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="en_PP-OCRv3_rec_infer" --rec_image_shape="3, 32, 100" --rec_char_dict_path="ppocr/utils/en_dict.txt"
python3 tools/infer/predict_system.py --image_dir="doc/imgs/ger_2.jpg" --det_model_dir="en_PP-OCRv3_det_infer" --rec_model_dir="en_PP-OCRv3_rec_infer" --rec_char_dict_path="ppocr/utils/en_dict.txt" --use_angle_cls=false 



