# HF Kien Fallback Source Audit

{
  "box_counts_by_raw_class_id": {
    "0": 3592,
    "1": 3343
  },
  "class_mapping": {
    "0": "fire",
    "1": "smoke"
  },
  "class_mapping_evidence": "Archive data.yaml uses literal names ['0', '1']; filename correlation showed fire filenames overwhelmingly label 0 and smoke filenames label 1, matching canonical 0=fire and 1=smoke.",
  "commercial_securevu_training": "allowed_for_R&D_pending_attribution_review",
  "corrupt_images": 0,
  "dataset_root": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke",
  "empty_labels": 0,
  "filename_label_signal": {
    "fire_and_smoke_filename": {
      "0": 9,
      "1": 2
    },
    "fire_filename": {
      "0": 1290,
      "1": 61
    },
    "other": {
      "0": 2293,
      "1": 3152
    },
    "smoke_filename": {
      "1": 128
    }
  },
  "invalid_label_files": 85,
  "invalid_label_handling": "excluded_before_prepare; not treated as negatives",
  "license_status": "HF metadata apache-2.0; archive data.yaml Roboflow license CC BY 4.0",
  "missing_labels": 0,
  "raw_class_names": {
    "0": "0",
    "1": "1"
  },
  "raw_image_counts_by_split": {
    "test": 750,
    "train": 3500,
    "valid": 750
  },
  "samples": {
    "corrupt": [],
    "invalid_labels": [
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/000023_jpg.rf.6c4a4869aecf6e7e97f9824d67d7543d.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/000023_jpg.rf.6c4a4869aecf6e7e97f9824d67d7543d.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/000697_jpg.rf.ec56de907717b66b3d9977601790e81f.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/000697_jpg.rf.ec56de907717b66b3d9977601790e81f.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/000714_jpg.rf.f0306f57cd281ac465916167d8918e0b.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/000714_jpg.rf.f0306f57cd281ac465916167d8918e0b.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/10_jpeg_jpg.rf.d5e85cf6e2d7fccbf928cd794987e6fa.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/10_jpeg_jpg.rf.d5e85cf6e2d7fccbf928cd794987e6fa.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/1537_jpg.rf.51f1e90b0d4f3d7517f474696f55c4d5.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/1537_jpg.rf.51f1e90b0d4f3d7517f474696f55c4d5.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/1538_jpg.rf.72bd0887348937b836ebcfbbf54defe6.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/1538_jpg.rf.72bd0887348937b836ebcfbbf54defe6.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/1_jpg.rf.c28ece532832ba8943b3ef67c9e539da.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/1_jpg.rf.c28ece532832ba8943b3ef67c9e539da.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/280aos_mp4-0100_jpg.rf.2ba458b8babf89111e6ea8d42040143f.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/280aos_mp4-0100_jpg.rf.2ba458b8babf89111e6ea8d42040143f.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/280aos_mp4-0107_jpg.rf.ca55fdf0159296b6d0c2395fc9a78b72.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/280aos_mp4-0107_jpg.rf.ca55fdf0159296b6d0c2395fc9a78b72.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/5-Valveglandleak_png.rf.415f5149dec37878298ff11818723b9b.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/5-Valveglandleak_png.rf.415f5149dec37878298ff11818723b9b.txt"
      },
      {
        "errors": [
          "line_3:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/777_harsh_jpg.rf.324f8cc57b44f85793336c55ed807f43.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/777_harsh_jpg.rf.324f8cc57b44f85793336c55ed807f43.txt"
      },
      {
        "errors": [
          "line_3:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/7test1914_jpg.rf.2ec6d34671c8f68840f20a2db0e9c689.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/7test1914_jpg.rf.2ec6d34671c8f68840f20a2db0e9c689.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/7test1930_jpg.rf.122d31e2e5f01c9b1b97ac30ccc52310.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/7test1930_jpg.rf.122d31e2e5f01c9b1b97ac30ccc52310.txt"
      },
      {
        "errors": [
          "line_3:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/8test3573_jpg.rf.e1cbd7236c28c80dc9a1c63913c291e5.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/8test3573_jpg.rf.e1cbd7236c28c80dc9a1c63913c291e5.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/8test4087_jpg.rf.581891d9972c05781328da4d0fbc67aa.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/8test4087_jpg.rf.581891d9972c05781328da4d0fbc67aa.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/8test8488_jpg.rf.bc531bdc3299e9f6e91cab6cbefb919c.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/8test8488_jpg.rf.bc531bdc3299e9f6e91cab6cbefb919c.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/ClosingYourBedroomDoorCouldSaveYourLife-KDKA2911_png_jpg.rf.8d8e06aed8c295f0a5f0859818869981.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/ClosingYourBedroomDoorCouldSaveYourLife-KDKA2911_png_jpg.rf.8d8e06aed8c295f0a5f0859818869981.txt"
      },
      {
        "errors": [
          "line_3:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/Inside-a-burning-house_135_jpg.rf.0236e4664acd2d59324c87e24240b3f0.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/Inside-a-burning-house_135_jpg.rf.0236e4664acd2d59324c87e24240b3f0.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/Smoke_img17_jpg.rf.b86bd5eb263b9a80230e60d0929b235e.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/Smoke_img17_jpg.rf.b86bd5eb263b9a80230e60d0929b235e.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/Smoke_img84_jpg.rf.d66fe9da8e7daa8289d426e411226dd3.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/Smoke_img84_jpg.rf.d66fe9da8e7daa8289d426e411226dd3.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/classk10_mp4-175_jpg.rf.4c6b1a4cb2b4f2aa2da0ec181e8c307f.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/classk10_mp4-175_jpg.rf.4c6b1a4cb2b4f2aa2da0ec181e8c307f.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/data_0000412_png.rf.516a6c3eed8138e800766857f1e386f9.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/data_0000412_png.rf.516a6c3eed8138e800766857f1e386f9.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/data_0000577_png.rf.73dd3f4090fff6990d3aa6181fb97d7d.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/data_0000577_png.rf.73dd3f4090fff6990d3aa6181fb97d7d.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/fire04630_jpg.rf.94350065605465036a66a458cc182164.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/fire04630_jpg.rf.94350065605465036a66a458cc182164.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/fire_roi11_jpg.rf.cfc9c7f97a63fc957d29d1faaff047d1.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/fire_roi11_jpg.rf.cfc9c7f97a63fc957d29d1faaff047d1.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000095_png.rf.18bbcfe45fbc8758df530ce76b0d526e.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000095_png.rf.18bbcfe45fbc8758df530ce76b0d526e.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000218_png.rf.95fadde638059feb65314a354a4b471a.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000218_png.rf.95fadde638059feb65314a354a4b471a.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000268_png.rf.8f7bcbb70199799ef339f436fff83b0d.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000268_png.rf.8f7bcbb70199799ef339f436fff83b0d.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000437_png.rf.4279672957d5c3df3ef70a0172273e5b.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000437_png.rf.4279672957d5c3df3ef70a0172273e5b.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000439_png.rf.024bc4d0100b68d17d77927f0fe9ab94.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000439_png.rf.024bc4d0100b68d17d77927f0fe9ab94.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000459_png.rf.675f169093207d0d031dac792de823f8.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000459_png.rf.675f169093207d0d031dac792de823f8.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000480_png.rf.c6a5e1b23fb715b16cd6d75f8fa3188d.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000480_png.rf.c6a5e1b23fb715b16cd6d75f8fa3188d.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000582_png.rf.f22f1a65527f931301c529995a6403c3.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000582_png.rf.f22f1a65527f931301c529995a6403c3.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000651_png.rf.7a5260280b60c7b510a1baa82788fd26.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000651_png.rf.7a5260280b60c7b510a1baa82788fd26.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000702_png.rf.ea26c7274dbc46177b04497f98db6435.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000702_png.rf.ea26c7274dbc46177b04497f98db6435.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000812_png.rf.06bcd025407c4b162a52e059ecdf9398.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000812_png.rf.06bcd025407c4b162a52e059ecdf9398.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000856_png.rf.1635cedcdb5e3ed6936667c4d54b8716.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000856_png.rf.1635cedcdb5e3ed6936667c4d54b8716.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000869_png.rf.ec906acbec663af3cb04ee54fcd5ee0d.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000869_png.rf.ec906acbec663af3cb04ee54fcd5ee0d.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0000959_png.rf.4b844b3b7cb7eb6cbc3e62e75d45f9b5.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0000959_png.rf.4b844b3b7cb7eb6cbc3e62e75d45f9b5.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001040_png.rf.634ff57a83194861ed11f887cfa10650.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001040_png.rf.634ff57a83194861ed11f887cfa10650.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001042_png.rf.a62bb5ba2fcf50e8fa3fe33532728f5e.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001042_png.rf.a62bb5ba2fcf50e8fa3fe33532728f5e.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001060_png.rf.f2eab0a7a3214e467e0c4a9321bab630.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001060_png.rf.f2eab0a7a3214e467e0c4a9321bab630.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001131_png.rf.626f0e8a87ee5c6addd8e4c3c253cbd7.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001131_png.rf.626f0e8a87ee5c6addd8e4c3c253cbd7.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001175_png.rf.35d650bbbb8df39410bf80b49b7ae7bd.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001175_png.rf.35d650bbbb8df39410bf80b49b7ae7bd.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0001179_png.rf.2568aac7c4e449cc2779210b95ab6fa2.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0001179_png.rf.2568aac7c4e449cc2779210b95ab6fa2.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_0002024_png.rf.96b1b9af2f6d66d88ce2143d03f8ed6c.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_0002024_png.rf.96b1b9af2f6d66d88ce2143d03f8ed6c.txt"
      },
      {
        "errors": [
          "line_3:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/frame_07_jpg.rf.5e8f3b8bac533375a0103e6de397f4da.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/frame_07_jpg.rf.5e8f3b8bac533375a0103e6de397f4da.txt"
      },
      {
        "errors": [
          "line_2:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/img_1030_jpg.rf.e4c2df05aaa9b9e5c2624f80958c10cd.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/img_1030_jpg.rf.e4c2df05aaa9b9e5c2624f80958c10cd.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/img_312_jpg.rf.c236695b214089515750a5d1a12341f2.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/img_312_jpg.rf.c236695b214089515750a5d1a12341f2.txt"
      },
      {
        "errors": [
          "line_1:box_outside_image"
        ],
        "image": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/images/img_317_jpg.rf.44284c423fa5d92dc5377aba9ed40b90.jpg",
        "label": "data/raw/hf_kien_fire_smoke/download/Indoor Fire Smoke/train/labels/img_317_jpg.rf.44284c423fa5d92dc5377aba9ed40b90.txt"
      }
    ],
    "missing_labels": []
  },
  "source": {
    "archive": "Indoor Fire Smoke.zip",
    "dataset": "KienNgyuen/Fire-Smoke-Detection",
    "url": "https://huggingface.co/datasets/KienNgyuen/Fire-Smoke-Detection"
  },
  "status": "PASS",
  "timestamp_utc": "2026-07-12T15:16:38.426150+00:00"
}
