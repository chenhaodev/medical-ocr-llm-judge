# Dataset Directory

This directory contains the medical OCR evaluation datasets.

## Structure

```
data/
├── reports/         # Medical laboratory reports
│   └── 使用的报告单/
│       ├── 1-23/    # Reports 1-23
│       ├── 24-40/   # Reports 24-40
│       └── ...      # Additional subdirectories
└── medicine/        # Medicine packaging images
    └── 药品-按药名分原图/
        ├── 美林 布洛芬混悬液/
        ├── 泰诺林/
        └── ...      # 98 medicine categories
```

## Dataset Details

### Medical Reports (101 images)
- **Type**: Blood test reports (血常规检验报告单)
- **Format**: JPG images
- **Content**:
  - Patient information
  - Test items and values
  - Reference ranges
  - Diagnostic symbols (↑, ↓, *, H, L, etc.)
  - Comments and annotations

### Medicine Packaging (261 images, 98 categories)
- **Type**: Medicine box photos
- **Format**: JPG images
- **Content**:
  - Medicine name
  - Specifications
  - Manufacturer
  - Approval number
  - Instructions

## Notes

- This directory is gitignored (datasets are large)
- Place your dataset here following the structure above
- The framework auto-detects document type based on path
