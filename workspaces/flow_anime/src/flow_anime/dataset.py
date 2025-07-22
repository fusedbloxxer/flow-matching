from .args import ConvertDatasetArgs, DatasetToolsArgs, DownloadDatasetArgs, EmbedDatasetArgs, EncodeDatasetArgs


def dataset(args: DatasetToolsArgs) -> None:
    match args.tool_args:
        case DownloadDatasetArgs():
            download(args.tool_args)
        case ConvertDatasetArgs():
            convert(args.tool_args)
        case EncodeDatasetArgs():
            encode(args.tool_args)
        case EmbedDatasetArgs():
            embed(args.tool_args)


def download(args: DownloadDatasetArgs) -> None:
    pass


def convert(args: ConvertDatasetArgs) -> None:
    pass


def encode(args: EncodeDatasetArgs) -> None:
    pass


def embed(args: EmbedDatasetArgs) -> None:
    pass
