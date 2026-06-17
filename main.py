"""
命令行工具 - 本地批量处理雷达数据
"""
import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.service import WeatherRadarService


def main():
    parser = argparse.ArgumentParser(description="短临天气推演平台 - 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    predict_parser = subparsers.add_parser("predict", help="处理雷达文件并生成预测")
    predict_parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="输入雷达文件或目录"
    )
    predict_parser.add_argument("-o", "--output", default="./output", help="输出目录")
    predict_parser.add_argument("--no-images", action="store_true", help="不生成图片")

    preview_parser = subparsers.add_parser("preview", help="生成单帧预览")
    preview_parser.add_argument("-f", "--file", required=True, help="雷达文件路径")
    preview_parser.add_argument("-v", "--var", default="Z", help="变量名 (Z, ZDR)")
    preview_parser.add_argument("-o", "--output", default="preview.png", help="输出图片路径")

    serve_parser = subparsers.add_parser("serve", help="启动 API 服务")
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


def cmd_predict(args):
    input_files = []
    for item in args.input:
        if os.path.isdir(item):
            for ext in ["*.bin", "*.dat", "*.bz2", "*"]:
                input_files.extend(glob.glob(os.path.join(item, ext)))
        elif os.path.isfile(item):
            input_files.append(item)

    input_files = [f for f in input_files if os.path.isfile(f)]
    input_files.sort()

    if not input_files:
        print("错误: 未找到输入文件")
        sys.exit(1)

    print(f"加载 {len(input_files)} 个雷达文件...")

    service = WeatherRadarService()
    result = service.process_files(
        input_files,
        save_images=not args.no_images,
        output_dir=args.output,
    )

    if not result["success"]:
        print(f"处理失败: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print(f"处理完成!")
    print(f"  输入帧数: {result['input_frame_count']}")
    print(f"  时间范围: {result['start_time']} ~ {result['end_time']}")
    print(f"  预测帧数: {result['output_frame_count']}")
    print(f"  预测起始: {result['prediction_start']}")
    print(f"  时间间隔: {result['prediction_interval_minutes']} 分钟")

    if "output_images" in result:
        print(f"  输出图片: {len(result['output_images'])} 个文件")
        for img_path in result["output_images"][:5]:
            print(f"    - {img_path}")
        if len(result["output_images"]) > 5:
            print(f"    ... 共 {len(result['output_images'])} 个")


def cmd_preview(args):
    if not os.path.exists(args.file):
        print(f"错误: 文件不存在 {args.file}")
        sys.exit(1)

    service = WeatherRadarService()
    img_bytes = service.get_single_preview(args.file, var_name=args.var)

    if img_bytes is None:
        print("错误: 生成预览失败")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(img_bytes)
    print(f"预览图已保存: {args.output}")


def cmd_serve(args):
    import uvicorn
    from app import create_app

    app = create_app()
    print(f"启动服务: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
