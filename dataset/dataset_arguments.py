from sup_func.sup_func import parse_left_args


def pathway_dataset_arguments(parser):
    parser.add_argument('--random_permutation', action='store_true')  # for the test set
    parser.add_argument('--subset_number', type=int, default=None)  # for the test set
    # parser.add_argument('--knowledge_ood', action='store_true')
    parser.add_argument('--subset_balance', action='store_true')
    parser.add_argument('--subcategory', type=str, default=None)
    parser.add_argument('--no_evaluation',
                        action='store_true')  # For reasoning tasks when the generation is going to be evaluated later.
    parser.add_argument('--evaluate_model', type=str, default='gpt-4o')
    return parser


def add_dataset_args(args, left):
    args, left_new = parse_left_args(args, left, pathway_dataset_arguments)
    args.task_type = 'judge' if 'judge' in args.dataset_name else 'reasoning'
    return args
