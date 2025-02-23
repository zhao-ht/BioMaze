from sup_func.sup_func import parse_left_args


def add_cot_args(parser):
    parser.add_argument("--in_context_num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_type", type=str)
    parser.add_argument("--enable_cot", action="store_true")
    return parser


def add_tog_args(parser):
    # parser.add_argument("--dataset", type=str,
    #                     default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--answer_type", type=str)
    parser.add_argument("--max_context_length", type=int)
    parser.add_argument("--answer_method", type=str)
    parser.add_argument("--remove_uncertainty", action="store_true")

    return parser


def add_cok_args(parser):
    parser.add_argument("--in_context_num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_type", type=str)
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--max_pieces", type=int)
    return parser


def add_graph_agent_args(parser):
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--memory_size", type=int)
    parser.add_argument("--answer_type", type=str)
    parser.add_argument("--answer_method", type=str)
    parser.add_argument("--remove_uncertainty", action="store_true")
    parser.add_argument("--uncertainty_query", action="store_true")
    parser.add_argument("--allow_history_cut", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cot_merge_method", type=str, default=None)
    return parser


def add_model_args(args, left):
    if args.planning_method in ['cot']:
        args, left_new = parse_left_args(args, left, add_cot_args)
    elif args.planning_method in ['tog']:
        args, left_new = parse_left_args(args, left, add_tog_args)
    elif args.planning_method in ['cok']:
        args, left_new = parse_left_args(args, left, add_cok_args)
    elif args.planning_method in ['graph_agent']:
        args, left_new = parse_left_args(args, left, add_graph_agent_args)
    else:
        raise ValueError("model not implied yet")

    return args
