from pdf2image import convert_from_path
from PIL import Image

def pdf_to_png(pdf_path, output_dir='./pdf_new', dpi=200, size=None):
    """
    PDF를 PNG로 변환합니다.

    Args:
        pdf_path (str): 변환할 PDF 파일 경로
        output_dir (str): 출력 PNG 파일을 저장할 폴더
        dpi (int): 해상도 (기본값 200)
        size (tuple[int, int] | None): (width, height) 형태로 출력 이미지 크기 지정 (옵션)
    """
    # PDF 페이지를 이미지로 변환
    pages = convert_from_path(pdf_path, dpi=dpi)

    output_paths = []
    for i, page in enumerate(pages):
        if size is not None:
            page = page.resize(size, Image.LANCZOS)

        output_path = f"{output_dir}/page_{i+1:03d}.png"
        page.save(output_path, 'PNG')
        output_paths.append(output_path)

    return output_paths

if __name__ == "__main__":
    paths = pdf_to_png(
        # pdf_path="2407.14170v1.pdf",
        pdf_path="2404.18399v2.pdf",
        
        output_dir="./png_new",
        dpi=1000,           # 해상도 조정
        size=(1700, 2200)  # 원하는 사이즈 지정 (픽셀 단위, 생략 가능)
    )
    print("변환 완료:", paths)