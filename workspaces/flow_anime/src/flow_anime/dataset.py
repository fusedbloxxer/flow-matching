from .args import DatasetConvertToolArgs, DatasetEncodeToolArgs, DatasetToolsArgs


def dataset(args: DatasetToolsArgs) -> None:
    match args.tool_args:
        case DatasetConvertToolArgs():
            convert(args.tool_args)
        case DatasetEncodeToolArgs():
            encode(args.tool_args)


def convert(args: DatasetConvertToolArgs) -> None:
    pass


def encode(args: DatasetEncodeToolArgs) -> None:
    pass
