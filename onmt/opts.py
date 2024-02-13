""" Implementation of all available options """
import configargparse

from onmt.modules.sru import CheckSRU
from onmt.transforms import AVAILABLE_TRANSFORMS
from onmt.schedulers import AVAILABLE_SCHEDULERS
from onmt.constants import ModelTask
from onmt.modules.position_ffn import ACTIVATION_FUNCTIONS
from onmt.modules.position_ffn import ActivationFunction
from onmt.constants import DefaultTokens


def config_opts(parser):
    group = parser.add_argument_group("Configuration")
    group.add(
        "-config",
        "--config",
        required=False,
        is_config_file_arg=True,
        help="Path of the main YAML config file.",
    )
    group.add(
        "-save_config",
        "--save_config",
        required=False,
        is_write_out_config_file_arg=True,
        help="Path where to save the config.",
    )


def _add_logging_opts(parser, is_train=True):
    group = parser.add_argument_group("Logging")
    group.add(
        "--log_file",
        "-log_file",
        type=str,
        default="",
        help="Output logs to a file under this path.",
    )
    group.add(
        "--log_file_level",
        "-log_file_level",
        type=str,
        action=StoreLoggingLevelAction,
        choices=StoreLoggingLevelAction.CHOICES,
        default="0",
    )
    group.add(
        "--verbose",
        "-verbose",
        action="store_true",
        help="Print data loading and statistics for all process"
        "(default only log the first process shard)"
        if is_train
        else "Print scores and predictions for each sentence",
    )

    if is_train:
        group.add(
            "--valid_metrics",
            "-valid_metrics",
            default=[],
            nargs="+",
            help="List of names of additional validation metrics",
        )
        group.add(
            "--scoring_debug",
            "-scoring_debug",
            action="store_true",
            help="Dump the src/ref/pred of the current batch",
        )
        group.add(
            "--dump_preds",
            "-dump_preds",
            type=str,
            default=None,
            help="Folder to dump predictions to.",
        )
        group.add(
            "--report_every",
            "-report_every",
            type=int,
            default=50,
            help="Print stats at this interval.",
        )
        group.add(
            "--exp_host",
            "-exp_host",
            type=str,
            default="",
            help="Send logs to this crayon server.",
        )
        group.add(
            "--exp",
            "-exp",
            type=str,
            default="",
            help="Name of the experiment for logging.",
        )
        # Use Tensorboard for visualization during training
        group.add(
            "--tensorboard",
            "-tensorboard",
            action="store_true",
            help="Use tensorboard for visualization during training. "
            "Must have the library tensorboard >= 1.14.",
        )
        group.add(
            "--tensorboard_log_dir",
            "-tensorboard_log_dir",
            type=str,
            default="runs/onmt",
            help="Log directory for Tensorboard. " "This is also the name of the run.",
        )
        group.add(
            "--override_opts",
            "-override-opts",
            action="store_true",
            help="Allow to override some checkpoint opts",
        )

        # Curriculum learning options
        group.add(
            "--curriculum_learning_enabled",
            "-curriculum_learning_enabled",
            type=bool,
            default=False,
        )
        group.add(
            "--curriculum_learning_steps",
            "-curriculum_learning_steps",
            type=int,
            default=100,
        )
        group.add(
            "--curriculum_learning_starting_task_id",
            "-curriculum_learning_starting_task_id",
            type=int,
            default=1,
        )
        group.add(
            "--curriculum_learning_scheduler",
            "-curriculum_learning_scheduler",
            type=str,
            default="alternation",
        )
        group.add(
            "--curriculum_learning_nb_tasks",
            "-curriculum_learning_nb_tasks",
            type=int,
            default=1,
        )

    else:
        # Options only during inference
        group.add(
            "--attn_debug",
            "-attn_debug",
            action="store_true",
            help="Print best attn for each word",
        )
        group.add(
            "--align_debug",
            "-align_debug",
            action="store_true",
            help="Print best align for each word",
        )
        group.add(
            "--dump_beam",
            "-dump_beam",
            type=str,
            default="",
            help="File to dump beam information to.",
        )
        group.add(
            "--n_best",
            "-n_best",
            type=int,
            default=1,
            help="If verbose is set, will output the n_best " "decoded sentences",
        )
        group.add(
            "--with_score",
            "-with_score",
            action="store_true",
            help="add a tab separated score to the translation",
        )


def _add_reproducibility_opts(parser):
    group = parser.add_argument_group("Reproducibility")
    group.add(
        "--seed",
        "-seed",
        type=int,
        default=-1,
        help="Set random seed used for better " "reproducibility between experiments.",
    )


def _add_dynamic_corpus_opts(parser, build_vocab_only=False):
    """Options related to training corpus, type: a list of dictionary."""
    group = parser.add_argument_group("Data")
    group.add(
        "-data",
        "--data",
        required=True,
        help="List of datasets and their specifications. "
        "See examples/*.yaml for further details.",
    )
    group.add(
        "-skip_empty_level",
        "--skip_empty_level",
        default="warning",
        choices=["silent", "warning", "error"],
        help="Security level when encounter empty examples."
        "silent: silently ignore/skip empty example;"
        "warning: warning when ignore/skip empty example;"
        "error: raise error & stop execution when encouter empty.",
    )
    group.add(
        "-transforms",
        "--transforms",
        default=[],
        nargs="+",
        choices=AVAILABLE_TRANSFORMS.keys(),
        help="Default transform pipeline to apply to data. "
        "Can be specified in each corpus of data to override.",
    )

    group.add(
        "-save_data",
        "--save_data",
        required=build_vocab_only,
        help="Output base path for objects that will "
        "be saved (vocab, transforms, embeddings, ...).",
    )
    group.add(
        "-overwrite",
        "--overwrite",
        action="store_true",
        help="Overwrite existing objects if any.",
    )
    group.add(
        "-n_sample",
        "--n_sample",
        type=int,
        default=(5000 if build_vocab_only else 0),
        help=("Build vocab using " if build_vocab_only else "Stop after save ")
        + "this number of transformed samples/corpus. Can be [-1, 0, N>0]. "
        "Set to -1 to go full corpus, 0 to skip.",
    )

    if not build_vocab_only:
        group.add(
            "-dump_transforms",
            "--dump_transforms",
            action="store_true",
            help="Dump transforms `*.transforms.pt` to disk."
            " -save_data should be set as saving prefix.",
        )
    else:
        group.add(
            "-dump_samples",
            "--dump_samples",
            action="store_true",
            help="Dump samples when building vocab. "
            "Warning: this may slow down the process.",
        )
        group.add(
            "-num_threads",
            "--num_threads",
            type=int,
            default=1,
            help="Number of parallel threads to build the vocab.",
        )
        group.add(
            "-learn_subwords",
            "--learn_subwords",
            action="store_true",
            help="Learn subwords prior to building vocab",
        )
        group.add(
            "-learn_subwords_size",
            "--learn_subwords_size",
            type=int,
            default=32000,
            help="Learn subwords operations",
        )
        group.add(
            "-vocab_sample_queue_size",
            "--vocab_sample_queue_size",
            type=int,
            default=20,
            help="Size of queues used in the build_vocab dump path.",
        )


def _add_features_opts(parser):
    group = parser.add_argument_group("Features")
    group.add(
        "-n_src_feats",
        "--n_src_feats",
        type=int,
        default=0,
        help="Number of source feats.",
    )
    group.add(
        "-src_feats_defaults",
        "--src_feats_defaults",
        help="Default features to apply in source in case " "there are not annotated",
    )


def _add_dynamic_vocab_opts(parser, build_vocab_only=False):
    """Options related to vocabulary and features.

    Add all options relate to vocabulary or features to parser.
    """
    group = parser.add_argument_group("Vocab")
    group.add(
        "-src_vocab",
        "--src_vocab",
        required=True,
        help=("Path to save" if build_vocab_only else "Path to")
        + " src (or shared) vocabulary file. "
        "Format: one <word> or <word>\t<count> per line.",
    )
    group.add(
        "-tgt_vocab",
        "--tgt_vocab",
        help=("Path to save" if build_vocab_only else "Path to")
        + " tgt vocabulary file. "
        "Format: one <word> or <word>\t<count> per line.",
    )
    group.add(
        "-share_vocab",
        "--share_vocab",
        action="store_true",
        help="Share source and target vocabulary.",
    )
    group.add(
        "--decoder_start_token",
        "-decoder_start_token",
        type=str,
        default=DefaultTokens.BOS,
        help="Default decoder start token "
        "for most ONMT models it is <s> = BOS "
        "it happens that for some Fairseq model it requires </s> ",
    )
    group.add(
        "--default_specials",
        "-default_specials",
        nargs="+",
        type=str,
        default=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ],
        help="default specials used for Vocab initialization"
        " UNK, PAD, BOS, EOS will take IDs 0, 1, 2, 3 "
        " typically <unk> <blank> <s> </s> ",
    )

    _add_features_opts(parser)

    if not build_vocab_only:
        group.add(
            "-src_vocab_size",
            "--src_vocab_size",
            type=int,
            default=32768,
            help="Maximum size of the source vocabulary.",
        )
        group.add(
            "-tgt_vocab_size",
            "--tgt_vocab_size",
            type=int,
            default=32768,
            help="Maximum size of the target vocabulary",
        )
        group.add(
            "-vocab_size_multiple",
            "--vocab_size_multiple",
            type=int,
            default=8,
            help="Make the vocabulary size a multiple of this value.",
        )

        group.add(
            "-src_words_min_frequency",
            "--src_words_min_frequency",
            type=int,
            default=0,
            help="Discard source words with lower frequency.",
        )
        group.add(
            "-tgt_words_min_frequency",
            "--tgt_words_min_frequency",
            type=int,
            default=0,
            help="Discard target words with lower frequency.",
        )

        # Truncation options, for text corpus
        group = parser.add_argument_group("Pruning")
        group.add(
            "--src_seq_length_trunc",
            "-src_seq_length_trunc",
            type=int,
            default=None,
            help="Truncate source sequence length.",
        )
        group.add(
            "--tgt_seq_length_trunc",
            "-tgt_seq_length_trunc",
            type=int,
            default=None,
            help="Truncate target sequence length.",
        )

        group = parser.add_argument_group("Embeddings")
        group.add(
            "-both_embeddings",
            "--both_embeddings",
            help="Path to the embeddings file to use "
            "for both source and target tokens.",
        )
        group.add(
            "-src_embeddings",
            "--src_embeddings",
            help="Path to the embeddings file to use for source tokens.",
        )
        group.add(
            "-tgt_embeddings",
            "--tgt_embeddings",
            help="Path to the embeddings file to use for target tokens.",
        )
        group.add(
            "-embeddings_type",
            "--embeddings_type",
            choices=["GloVe", "word2vec"],
            help="Type of embeddings file.",
        )


def _add_dynamic_transform_opts(parser):
    """Options related to transforms.

    Options that specified in the definitions of each transform class
    at `onmt/transforms/*.py`.
    """
    for name, transform_cls in AVAILABLE_TRANSFORMS.items():
        transform_cls.add_options(parser)


def _add_dynamic_curriculum_opts(parser):
    """Options related to curriculum learning."""
    for name, transform_cls in AVAILABLE_SCHEDULERS.items():
        transform_cls.add_options(parser)


def dynamic_prepare_opts(parser, build_vocab_only=False):
    """Options related to data prepare in dynamic mode.

    Add all dynamic data prepare related options to parser.
    If `build_vocab_only` set to True, then only contains options that
    will be used in `onmt/bin/build_vocab.py`.
    """
    config_opts(parser)
    _add_dynamic_corpus_opts(parser, build_vocab_only=build_vocab_only)
    _add_dynamic_vocab_opts(parser, build_vocab_only=build_vocab_only)
    _add_dynamic_transform_opts(parser)
    _add_dynamic_curriculum_opts(parser)

    if build_vocab_only:
        _add_reproducibility_opts(parser)
        # as for False, this will be added in _add_train_general_opts


def distributed_opts(parser):
    # GPU
    group = parser.add_argument_group("Distributed")
    group.add(
        "--gpu_ranks",
        "-gpu_ranks",
        default=[],
        nargs="*",
        type=int,
        help="list of ranks of each process.",
    )
    group.add(
        "--world_size",
        "-world_size",
        default=1,
        type=int,
        help="total number of distributed processes.",
    )
    group.add(
        "--parallel_mode",
        "-parallel_mode",
        default="data_parallel",
        choices=["tensor_parallel", "data_parallel"],
        type=str,
        help="Distributed mode.",
    )
    group.add(
        "--gpu_backend",
        "-gpu_backend",
        default="nccl",
        type=str,
        help="Type of torch distributed backend",
    )
    group.add(
        "--gpu_verbose_level",
        "-gpu_verbose_level",
        default=0,
        type=int,
        help="Gives more info on each process per GPU.",
    )
    group.add(
        "--master_ip",
        "-master_ip",
        default="localhost",
        type=str,
        help="IP of master for torch.distributed training.",
    )
    group.add(
        "--master_port",
        "-master_port",
        default=10000,
        type=int,
        help="Port of master for torch.distributed training.",
    )


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group("Model-Embeddings")
    group.add(
        "--src_word_vec_size",
        "-src_word_vec_size",
        type=int,
        default=500,
        help="Word embedding size for src.",
    )
    group.add(
        "--tgt_word_vec_size",
        "-tgt_word_vec_size",
        type=int,
        default=500,
        help="Word embedding size for tgt.",
    )
    group.add(
        "--word_vec_size",
        "-word_vec_size",
        type=int,
        default=-1,
        help="Word embedding size for src and tgt.",
    )

    group.add(
        "--share_decoder_embeddings",
        "-share_decoder_embeddings",
        action="store_true",
        help="Use a shared weight matrix for the input and "
        "output word  embeddings in the decoder.",
    )
    group.add(
        "--share_embeddings",
        "-share_embeddings",
        action="store_true",
        help="Share the word embeddings between encoder "
        "and decoder. Need to use shared dictionary for this "
        "option.",
    )
    group.add(
        "--position_encoding",
        "-position_encoding",
        action="store_true",
        help="Use a sin to mark relative words positions. "
        "Necessary for non-RNN style models.",
    )
    group.add(
        "--position_encoding_type",
        "-position_encoding_type",
        type=str,
        default="SinusoidalInterleaved",
        choices=["SinusoidalInterleaved", "SinusoidalConcat"],
        help="Type of positional encoding. At the moment: "
        "Sinusoidal fixed, Interleaved or Concat",
    )

    group.add(
        "-update_vocab",
        "--update_vocab",
        action="store_true",
        help="Update source and target existing vocabularies",
    )

    group = parser.add_argument_group("Model-Embedding Features")
    group.add(
        "--feat_merge",
        "-feat_merge",
        type=str,
        default="concat",
        choices=["concat", "sum", "mlp"],
        help="Merge action for incorporating features embeddings. "
        "Options [concat|sum|mlp].",
    )
    group.add(
        "--feat_vec_size",
        "-feat_vec_size",
        type=int,
        default=-1,
        help="If specified, feature embedding sizes "
        "will be set to this. Otherwise, feat_vec_exponent "
        "will be used.",
    )
    group.add(
        "--feat_vec_exponent",
        "-feat_vec_exponent",
        type=float,
        default=0.7,
        help="If -feat_merge_size is not set, feature "
        "embedding sizes will be set to N^feat_vec_exponent "
        "where N is the number of values the feature takes.",
    )

    # Model Task Options
    group = parser.add_argument_group("Model- Task")
    group.add(
        "-model_task",
        "--model_task",
        default=ModelTask.SEQ2SEQ,
        choices=[ModelTask.SEQ2SEQ, ModelTask.LANGUAGE_MODEL],
        help="Type of task for the model either seq2seq or lm",
    )

    # Encoder-Decoder Options
    group = parser.add_argument_group("Model- Encoder-Decoder")
    group.add(
        "--model_type",
        "-model_type",
        default="text",
        choices=["text"],
        help="Type of source model to use. Allows "
        "the system to incorporate non-text inputs. "
        "Options are [text].",
    )
    group.add(
        "--model_dtype",
        "-model_dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Data type of the model.",
    )

    group.add(
        "--encoder_type",
        "-encoder_type",
        type=str,
        default="rnn",
        help="Type of encoder layer to use. Non-RNN layers "
        "are experimental. Default options are "
        "[rnn|brnn|ggnn|mean|transformer|cnn|transformer_lm].",
    )
    group.add(
        "--decoder_type",
        "-decoder_type",
        type=str,
        default="rnn",
        help="Type of decoder layer to use. Non-RNN layers "
        "are experimental. Default options are "
        "[rnn|transformer|cnn|transformer].",
    )

    # Freeze Encoder and/or Decoder
    group.add(
        "--freeze_encoder",
        "-freeze_encoder",
        action="store_true",
        help="Freeze parameters in encoder.",
    )
    group.add(
        "--freeze_decoder",
        "-freeze_decoder",
        action="store_true",
        help="Freeze parameters in decoder.",
    )

    group.add(
        "--layers", "-layers", type=int, default=-1, help="Number of layers in enc/dec."
    )
    group.add(
        "--enc_layers",
        "-enc_layers",
        type=int,
        default=2,
        help="Number of layers in the encoder",
    )
    group.add(
        "--dec_layers",
        "-dec_layers",
        type=int,
        default=2,
        help="Number of layers in the decoder",
    )
    group.add(
        "--hidden_size",
        "-hidden_size",
        type=int,
        default=-1,
        help="Size of rnn hidden states. Overwrites " "enc_hid_size and dec_hid_size",
    )
    group.add(
        "--enc_hid_size",
        "-enc_hid_size",
        type=int,
        default=500,
        help="Size of encoder rnn hidden states.",
    )
    group.add(
        "--dec_hid_size",
        "-dec_hid_size",
        type=int,
        default=500,
        help="Size of decoder rnn hidden states.",
    )
    group.add(
        "--cnn_kernel_width",
        "-cnn_kernel_width",
        type=int,
        default=3,
        help="Size of windows in the cnn, the kernel_size is "
        "(cnn_kernel_width, 1) in conv layer",
    )

    group.add(
        "--layer_norm",
        "-layer_norm",
        type=str,
        default="standard",
        choices=["standard", "rms"],
        help="The type of layer"
        " normalization in the transformer architecture. Choices are"
        " standard or rms. Default to standard",
    )
    group.add(
        "--norm_eps", "-norm_eps", type=float, default=1e-6, help="Layer norm epsilon"
    )

    group.add(
        "--pos_ffn_activation_fn",
        "-pos_ffn_activation_fn",
        type=str,
        default=ActivationFunction.relu,
        choices=ACTIVATION_FUNCTIONS.keys(),
        help="The activation"
        " function to use in PositionwiseFeedForward layer. Choices are"
        f" {ACTIVATION_FUNCTIONS.keys()}. Default to"
        f" {ActivationFunction.relu}.",
    )

    group.add(
        "--input_feed",
        "-input_feed",
        type=int,
        default=1,
        help="Feed the context vector at each time step as "
        "additional input (via concatenation with the word "
        "embeddings) to the decoder.",
    )
    group.add(
        "--bridge",
        "-bridge",
        action="store_true",
        help="Have an additional layer between the last encoder "
        "state and the first decoder state",
    )
    group.add(
        "--rnn_type",
        "-rnn_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU", "SRU"],
        action=CheckSRU,
        help="The gate type to use in the RNNs",
    )
    group.add(
        "--context_gate",
        "-context_gate",
        type=str,
        default=None,
        choices=["source", "target", "both"],
        help="Type of context gate to use. " "Do not select for no context gate.",
    )

    # The following options (bridge_extra_node to n_steps) are used
    # for training with --encoder_type ggnn (Gated Graph Neural Network).
    group.add(
        "--bridge_extra_node",
        "-bridge_extra_node",
        type=bool,
        default=True,
        help="Graph encoder bridges only extra node to decoder as input",
    )
    group.add(
        "--bidir_edges",
        "-bidir_edges",
        type=bool,
        default=True,
        help="Graph encoder autogenerates bidirectional edges",
    )
    group.add(
        "--state_dim",
        "-state_dim",
        type=int,
        default=512,
        help="Number of state dimensions in the graph encoder",
    )
    group.add(
        "--n_edge_types",
        "-n_edge_types",
        type=int,
        default=2,
        help="Number of edge types in the graph encoder",
    )
    group.add(
        "--n_node",
        "-n_node",
        type=int,
        default=2,
        help="Number of nodes in the graph encoder",
    )
    group.add(
        "--n_steps",
        "-n_steps",
        type=int,
        default=2,
        help="Number of steps to advance graph encoder",
    )
    group.add(
        "--src_ggnn_size",
        "-src_ggnn_size",
        type=int,
        default=0,
        help="Vocab size plus feature space for embedding input",
    )

    # Attention options
    group = parser.add_argument_group("Model- Attention")
    group.add(
        "--global_attention",
        "-global_attention",
        type=str,
        default="general",
        choices=["dot", "general", "mlp", "none"],
        help="The attention type to use: "
        "dotprod or general (Luong) or MLP (Bahdanau)",
    )
    group.add(
        "--global_attention_function",
        "-global_attention_function",
        type=str,
        default="softmax",
        choices=["softmax", "sparsemax"],
    )
    group.add(
        "--self_attn_type",
        "-self_attn_type",
        type=str,
        default="scaled-dot",
        help="Self attention type in Transformer decoder "
        'layer -- currently "scaled-dot" or "average" ',
    )
    group.add(
        "--max_relative_positions",
        "-max_relative_positions",
        type=int,
        default=0,
        help="This setting enable relative position encoding"
        "We support two types of encodings:"
        "set this -1 to enable Rotary Embeddings"
        "more info: https://arxiv.org/abs/2104.09864"
        "set this to > 0 (ex: 16, 32) to use"
        "Maximum distance between inputs in relative "
        "positions representations. "
        "more info: https://arxiv.org/pdf/1803.02155.pdf",
    )
    group.add(
        "--relative_positions_buckets",
        "-relative_positions_buckets",
        type=int,
        default=0,
        help="This setting enable relative position bias"
        "more info: https://github.com/google-research/text-to-text-transfer-transformer",
    )
    group.add(
        "--rotary_interleave",
        "-rotary_interleave",
        type=bool,
        default=True,
        help="Interleave the head dimensions when rotary"
        " embeddings are applied."
        "    Otherwise the head dimensions are sliced in half."
        "True = default Llama from Meta (original)"
        "False = used by all Hugging face models",
    )
    group.add(
        "--heads",
        "-heads",
        type=int,
        default=8,
        help="Number of heads for transformer self-attention",
    )
    group.add(
        "--sliding_window",
        "-sliding_window",
        type=int,
        default=0,
        help="sliding window for transformer self-attention",
    )
    group.add(
        "--transformer_ff",
        "-transformer_ff",
        type=int,
        default=2048,
        help="Size of hidden transformer feed-forward",
    )
    group.add(
        "--aan_useffn",
        "-aan_useffn",
        action="store_true",
        help="Turn on the FFN layer in the AAN decoder",
    )
    group.add(
        "--add_qkvbias",
        "-add_qkvbias",
        action="store_true",
        help="Add bias to nn.linear of Query/Key/Value in MHA"
        "Note: this will add bias to output proj layer too",
    )
    group.add(
        "--multiquery",
        "-multiquery",
        action="store_true",
        help="Use MultiQuery attention" "Note: https://arxiv.org/pdf/1911.02150.pdf",
    )
    group.add(
        "--num_kv",
        "-num_kv",
        type=int,
        default=0,
        help="Number of heads for KV in the variant of MultiQuery attention (egs: Falcon 40B)",
    )
    group.add(
        "--add_ffnbias",
        "-add_ffnbias",
        action="store_true",
        help="Add bias to nn.linear of Position_wise FFN",
    )
    group.add(
        "--parallel_residual",
        "-parallel_residual",
        action="store_true",
        help="Use Parallel residual in Decoder Layer"
        "Note: this is used by GPT-J / Falcon Architecture",
    )
    group.add(
        "--shared_layer_norm",
        "-shared_layer_norm",
        action="store_true",
        help="Use a shared layer_norm in parallel residual attention"
        "Note: must be true for Falcon 7B / false for Falcon 40B"
        "same for GPT-J and GPT-NeoX models",
    )
    # Alignement options
    group = parser.add_argument_group("Model - Alignement")
    group.add(
        "--lambda_align",
        "-lambda_align",
        type=float,
        default=0.0,
        help="Lambda value for alignement loss of Garg et al (2019)"
        "For more detailed information, see: "
        "https://arxiv.org/abs/1909.02074",
    )
    group.add(
        "--alignment_layer",
        "-alignment_layer",
        type=int,
        default=-3,
        help="Layer number which has to be supervised.",
    )
    group.add(
        "--alignment_heads",
        "-alignment_heads",
        type=int,
        default=0,
        help="N. of cross attention heads per layer to supervised with",
    )
    group.add(
        "--full_context_alignment",
        "-full_context_alignment",
        action="store_true",
        help="Whether alignment is conditioned on full target context.",
    )

    # Generator and loss options.
    group = parser.add_argument_group("Generator")
    group.add(
        "--copy_attn",
        "-copy_attn",
        action="store_true",
        help="Train copy attention layer.",
    )
    group.add(
        "--copy_attn_type",
        "-copy_attn_type",
        type=str,
        default=None,
        choices=["dot", "general", "mlp", "none"],
        help="The copy attention type to use. Leave as None to use "
        "the same as -global_attention.",
    )
    group.add(
        "--generator_function",
        "-generator_function",
        default="softmax",
        choices=["softmax", "sparsemax"],
        help="Which function to use for generating "
        "probabilities over the target vocabulary (choices: "
        "softmax, sparsemax)",
    )
    group.add(
        "--copy_attn_force",
        "-copy_attn_force",
        action="store_true",
        help="When available, train to copy.",
    )
    group.add(
        "--reuse_copy_attn",
        "-reuse_copy_attn",
        action="store_true",
        help="Reuse standard attention for copy",
    )
    group.add(
        "--copy_loss_by_seqlength",
        "-copy_loss_by_seqlength",
        action="store_true",
        help="Divide copy loss by length of sequence",
    )
    group.add(
        "--coverage_attn",
        "-coverage_attn",
        action="store_true",
        help="Train a coverage attention layer.",
    )
    group.add(
        "--lambda_coverage",
        "-lambda_coverage",
        type=float,
        default=0.0,
        help="Lambda value for coverage loss of See et al (2017)",
    )
    group.add(
        "--lm_prior_model",
        "-lm_prior_model",
        type=str,
        default=None,
        help="LM model to used to train the TM",
    )
    group.add(
        "--lm_prior_lambda",
        "-lambda_prior_lambda",
        type=float,
        default=0.0,
        help="LM Prior Lambda",
    )
    group.add(
        "--lm_prior_tau",
        "-lambda_prior_tau",
        type=float,
        default=1.0,
        help="LM Prior Tau",
    )
    group.add(
        "--loss_scale",
        "-loss_scale",
        type=float,
        default=0,
        help="For FP16 training, the static loss scale to use. If not "
        "set, the loss scale is dynamically computed.",
    )
    group.add(
        "--apex_opt_level",
        "-apex_opt_level",
        type=str,
        default="",
        choices=["", "O0", "O1", "O2", "O3"],
        help="For FP16 training, the opt_level to use."
        "See https://nvidia.github.io/apex/amp.html#opt-levels.",
    )
    group.add(
        "--zero_out_prompt_loss",
        "-zero_out_prompt_loss",
        action="store_true",
        help="Set the prompt loss to zero."
        "Mostly for LLM finetuning."
        "Will be enabled only if the `insert_mask_before_placeholder` transform is applied",
    )
    group.add(
        "--use_ckpting",
        "-use_ckpting",
        default=[],
        nargs="+",
        choices=["ffn", "mha", "lora"],
        type=str,
        help="use gradient checkpointing those modules",
    )


def _add_train_general_opts(parser):
    """General options for training"""
    group = parser.add_argument_group("General")
    group.add(
        "--data_type",
        "-data_type",
        default="text",
        help="Type of the source input. " "Options are [text].",
    )

    group.add(
        "--save_model",
        "-save_model",
        default="model",
        help="Model filename (the model will be saved as "
        "<save_model>_N.pt where N is the number "
        "of steps",
    )

    group.add(
        "--save_format",
        "-save_format",
        default="pytorch",
        choices=["pytorch", "safetensors"],
        help="Format to save the model weights",
    )

    group.add(
        "--save_checkpoint_steps",
        "-save_checkpoint_steps",
        type=int,
        default=5000,
        help="""Save a checkpoint every X steps""",
    )
    group.add(
        "--keep_checkpoint",
        "-keep_checkpoint",
        type=int,
        default=-1,
        help="Keep X checkpoints (negative: keep all)",
    )

    # LoRa
    group.add(
        "--lora_layers",
        "-lora_layers",
        default=[],
        nargs="+",
        type=str,
        help="list of layers to be replaced by LoRa layers."
        " ex: ['linear_values', 'linear_query'] "
        " cf paper §4.2 https://arxiv.org/abs/2106.09685",
    )
    group.add(
        "--lora_embedding",
        "-lora_embedding",
        action="store_true",
        help="replace embeddings with LoRa Embeddings see §5.1",
    )
    group.add(
        "--lora_rank",
        "-lora_rank",
        type=int,
        default=2,
        help="r=2 successfully tested with NLLB-200 3.3B",
    )
    group.add(
        "--lora_alpha",
        "-lora_alpha",
        type=int,
        default=1,
        help="§4.1 https://arxiv.org/abs/2106.09685",
    )
    group.add(
        "--lora_dropout",
        "-lora_dropout",
        type=float,
        default=0.0,
        help="rule of thumb: same value as in main model",
    )

    _add_reproducibility_opts(parser)

    # Init options
    group = parser.add_argument_group("Initialization")
    group.add(
        "--param_init",
        "-param_init",
        type=float,
        default=0.1,
        help="Parameters are initialized over uniform distribution "
        "with support (-param_init, param_init). "
        "Use 0 to not use initialization",
    )
    group.add(
        "--param_init_glorot",
        "-param_init_glorot",
        action="store_true",
        help="Init parameters with xavier_uniform. " "Required for transformer.",
    )

    group.add(
        "--train_from",
        "-train_from",
        default="",
        type=str,
        help="If training from a checkpoint then this is the "
        "path to the pretrained model's state_dict.",
    )
    group.add(
        "--reset_optim",
        "-reset_optim",
        default="none",
        choices=["none", "all", "states", "keep_states"],
        help="Optimization resetter when train_from.",
    )

    # Pretrained word vectors
    group.add(
        "--pre_word_vecs_enc",
        "-pre_word_vecs_enc",
        help="If a valid path is specified, then this will load "
        "pretrained word embeddings on the encoder side. "
        "See README for specific formatting instructions.",
    )
    group.add(
        "--pre_word_vecs_dec",
        "-pre_word_vecs_dec",
        help="If a valid path is specified, then this will load "
        "pretrained word embeddings on the decoder side. "
        "See README for specific formatting instructions.",
    )
    # Freeze word vectors
    group.add(
        "--freeze_word_vecs_enc",
        "-freeze_word_vecs_enc",
        action="store_true",
        help="Freeze word embeddings on the encoder side.",
    )
    group.add(
        "--freeze_word_vecs_dec",
        "-freeze_word_vecs_dec",
        action="store_true",
        help="Freeze word embeddings on the decoder side.",
    )

    # Optimization options
    group = parser.add_argument_group("Optimization- Type")
    group.add(
        "--num_workers",
        "-num_workers",
        type=int,
        default=2,
        help="pytorch DataLoader num_workers",
    )
    group.add(
        "--batch_size",
        "-batch_size",
        type=int,
        default=64,
        help="Maximum batch size for training",
    )
    group.add(
        "--batch_size_multiple",
        "-batch_size_multiple",
        type=int,
        default=1,
        help="Batch size multiple for token batches.",
    )
    group.add(
        "--batch_type",
        "-batch_type",
        default="sents",
        choices=["sents", "tokens"],
        help="Batch grouping for batch_size. Standard "
        "is sents. Tokens will do dynamic batching",
    )
    group.add(
        "--normalization",
        "-normalization",
        default="sents",
        choices=["sents", "tokens"],
        help="Normalization method of the gradient.",
    )
    group.add(
        "--accum_count",
        "-accum_count",
        type=int,
        nargs="+",
        default=[1],
        help="Accumulate gradient this many times. "
        "Approximately equivalent to updating "
        "batch_size * accum_count batches at once. "
        "Recommended for Transformer.",
    )
    group.add(
        "--accum_steps",
        "-accum_steps",
        type=int,
        nargs="+",
        default=[0],
        help="Steps at which accum_count values change",
    )
    group.add(
        "--valid_steps",
        "-valid_steps",
        type=int,
        default=10000,
        help="Perfom validation every X steps",
    )
    group.add(
        "--valid_batch_size",
        "-valid_batch_size",
        type=int,
        default=32,
        help="Maximum batch size for validation",
    )
    group.add(
        "--train_steps",
        "-train_steps",
        type=int,
        default=100000,
        help="Number of training steps",
    )
    group.add(
        "--single_pass",
        "-single_pass",
        action="store_true",
        help="Make a single pass over the training dataset.",
    )
    group.add(
        "--early_stopping",
        "-early_stopping",
        type=int,
        default=0,
        help="Number of validation steps without improving.",
    )
    group.add(
        "--early_stopping_criteria",
        "-early_stopping_criteria",
        nargs="*",
        default=None,
        help="Criteria to use for early stopping.",
    )
    group.add(
        "--optim",
        "-optim",
        default="sgd",
        choices=[
            "sgd",
            "adagrad",
            "adadelta",
            "adam",
            "sparseadam",
            "adafactor",
            "fusedadam",
            "adamw8bit",
            "pagedadamw8bit",
            "pagedadamw32bit",
        ],
        help="Optimization method.",
    )
    group.add(
        "--adagrad_accumulator_init",
        "-adagrad_accumulator_init",
        type=float,
        default=0,
        help="Initializes the accumulator values in adagrad. "
        "Mirrors the initial_accumulator_value option "
        "in the tensorflow adagrad (use 0.1 for their default).",
    )
    group.add(
        "--max_grad_norm",
        "-max_grad_norm",
        type=float,
        default=5,
        help="If the norm of the gradient vector exceeds this, "
        "renormalize it to have the norm equal to "
        "max_grad_norm",
    )
    group.add(
        "--dropout",
        "-dropout",
        type=float,
        default=[0.3],
        nargs="+",
        help="Dropout probability; applied in LSTM stacks.",
    )
    group.add(
        "--attention_dropout",
        "-attention_dropout",
        type=float,
        default=[0.1],
        nargs="+",
        help="Attention Dropout probability.",
    )
    group.add(
        "--dropout_steps",
        "-dropout_steps",
        type=int,
        nargs="+",
        default=[0],
        help="Steps at which dropout changes.",
    )
    group.add(
        "--truncated_decoder",
        "-truncated_decoder",
        type=int,
        default=0,
        help="""Truncated bptt.""",
    )
    group.add(
        "--adam_beta1",
        "-adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter used by Adam. "
        "Almost without exception a value of 0.9 is used in "
        "the literature, seemingly giving good results, "
        "so we would discourage changing this value from "
        "the default without due consideration.",
    )
    group.add(
        "--adam_beta2",
        "-adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter used by Adam. "
        "Typically a value of 0.999 is recommended, as this is "
        "the value suggested by the original paper describing "
        "Adam, and is also the value adopted in other frameworks "
        "such as Tensorflow and Keras, i.e. see: "
        "https://www.tensorflow.org/api_docs/python/tf/train/Adam"
        "Optimizer or https://keras.io/optimizers/ . "
        'Whereas recently the paper "Attention is All You Need" '
        "suggested a value of 0.98 for beta2, this parameter may "
        "not work well for normal models / default "
        "baselines.",
    )
    group.add(
        "--label_smoothing",
        "-label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value epsilon. "
        "Probabilities of all non-true labels "
        "will be smoothed by epsilon / (vocab_size - 1). "
        "Set to zero to turn off label smoothing. "
        "For more detailed information, see: "
        "https://arxiv.org/abs/1512.00567",
    )
    group.add(
        "--average_decay",
        "-average_decay",
        type=float,
        default=0,
        help="Moving average decay. "
        "Set to other than 0 (e.g. 1e-4) to activate. "
        "Similar to Marian NMT implementation: "
        "http://www.aclweb.org/anthology/P18-4020 "
        "For more detail on Exponential Moving Average: "
        "https://en.wikipedia.org/wiki/Moving_average",
    )
    group.add(
        "--average_every",
        "-average_every",
        type=int,
        default=1,
        help="Step for moving average. "
        "Default is every update, "
        "if -average_decay is set.",
    )

    # learning rate
    group = parser.add_argument_group("Optimization- Rate")
    group.add(
        "--learning_rate",
        "-learning_rate",
        type=float,
        default=1.0,
        help="Starting learning rate. "
        "Recommended settings: sgd = 1, adagrad = 0.1, "
        "adadelta = 1, adam = 0.001",
    )
    group.add(
        "--learning_rate_decay",
        "-learning_rate_decay",
        type=float,
        default=0.5,
        help="If update_learning_rate, decay learning rate by "
        "this much if steps have gone past "
        "start_decay_steps",
    )
    group.add(
        "--start_decay_steps",
        "-start_decay_steps",
        type=int,
        default=50000,
        help="Start decaying every decay_steps after " "start_decay_steps",
    )
    group.add(
        "--decay_steps",
        "-decay_steps",
        type=int,
        default=10000,
        help="Decay every decay_steps",
    )

    group.add(
        "--decay_method",
        "-decay_method",
        type=str,
        default="none",
        choices=["noam", "noamwd", "rsqrt", "none"],
        help="Use a custom decay rate.",
    )
    group.add(
        "--warmup_steps",
        "-warmup_steps",
        type=int,
        default=4000,
        help="Number of warmup steps for custom decay.",
    )
    _add_logging_opts(parser, is_train=True)


def _add_train_dynamic_data(parser):
    group = parser.add_argument_group("Dynamic data")
    group.add(
        "-bucket_size",
        "--bucket_size",
        type=int,
        default=262144,
        help="""A bucket is a buffer of bucket_size examples to pick
                   from the various Corpora. The dynamic iterator batches
                   batch_size batchs from the bucket and shuffle them.""",
    )
    group.add(
        "-bucket_size_init",
        "--bucket_size_init",
        type=int,
        default=-1,
        help="""The bucket is initalized with this awith this
               amount of examples (optional)""",
    )
    group.add(
        "-bucket_size_increment",
        "--bucket_size_increment",
        type=int,
        default=0,
        help="""The bucket size is incremented with this
              amount of examples (optional)""",
    )
    group.add(
        "-prefetch_factor",
        "--prefetch_factor",
        type=int,
        default=200,
        help="""number of mini-batches loaded in advance to avoid the
                   GPU waiting during the refilling of the bucket.""",
    )


def _add_quant_opts(parser):
    group = parser.add_argument_group("Quant options")
    group.add(
        "--quant_layers",
        "-quant_layers",
        default=[],
        nargs="+",
        type=str,
        help="list of layers to be compressed in 4/8bit.",
    )

    group.add(
        "--quant_type",
        "-quant_type",
        default="",
        choices=[
            "",
            "bnb_8bit",
            "bnb_FP4",
            "bnb_NF4",
            "llm_awq",
            "aawq_gemm",
            "aawq_gemv",
        ],
        type=str,
        help="Type of compression.",
    )
    group.add(
        "--w_bit",
        "-w_bit",
        type=int,
        default=4,
        choices=[4],
        help="W_bit quantization.",
    )
    group.add(
        "--group_size",
        "-group_size",
        default=128,
        choices=[128],
        type=int,
        help="group size quantization.",
    )


def train_opts(parser):
    """All options used in train."""
    # options relate to data preprare
    dynamic_prepare_opts(parser, build_vocab_only=False)
    distributed_opts(parser)
    # options relate to train
    model_opts(parser)
    _add_train_general_opts(parser)
    _add_train_dynamic_data(parser)
    _add_quant_opts(parser)


def _add_decoding_opts(parser):
    group = parser.add_argument_group("Beam Search")
    beam_size = group.add(
        "--beam_size", "-beam_size", type=int, default=5, help="Beam size"
    )
    group.add(
        "--ratio",
        "-ratio",
        type=float,
        default=-0.0,
        help="Ratio based beam stop condition",
    )

    group = parser.add_argument_group("Random Sampling")
    group.add(
        "--random_sampling_topk",
        "-random_sampling_topk",
        default=0,
        type=int,
        help="Set this to -1 to do random sampling from full "
        "distribution. Set this to value k>1 to do random "
        "sampling restricted to the k most likely next tokens. "
        "Set this to 1 to use argmax.",
    )
    group.add(
        "--random_sampling_topp",
        "-random_sampling_topp",
        default=0.0,
        type=float,
        help="Probability for top-p/nucleus sampling. Restrict tokens"
        " to the most likely until the cumulated probability is"
        " over p. In range [0, 1]."
        " https://arxiv.org/abs/1904.09751",
    )
    group.add(
        "--random_sampling_temp",
        "-random_sampling_temp",
        default=1.0,
        type=float,
        help="If doing random sampling, divide the logits by "
        "this before computing softmax during decoding.",
    )
    group._group_actions.append(beam_size)
    _add_reproducibility_opts(parser)

    group = parser.add_argument_group(
        "Penalties", ".. Note:: Coverage Penalty is not available in sampling."
    )
    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    # Length penalty options
    group.add(
        "--length_penalty",
        "-length_penalty",
        default="avg",
        choices=["none", "wu", "avg"],
        help="Length Penalty to use.",
    )
    group.add(
        "--alpha",
        "-alpha",
        type=float,
        default=1.0,
        help="Length penalty parameter" "(higher = longer generation)",
    )
    # Coverage penalty options
    group.add(
        "--coverage_penalty",
        "-coverage_penalty",
        default="none",
        choices=["none", "wu", "summary"],
        help="Coverage Penalty to use. Only available in beam search.",
    )
    group.add(
        "--beta", "-beta", type=float, default=-0.0, help="Coverage penalty parameter"
    )
    group.add(
        "--stepwise_penalty",
        "-stepwise_penalty",
        action="store_true",
        help="Apply coverage penalty at every decoding step. "
        "Helpful for summary penalty.",
    )

    group = parser.add_argument_group(
        "Decoding tricks",
        ".. Tip:: Following options can be used to limit the decoding length "
        "or content.",
    )
    # Decoding Length constraint
    group.add(
        "--min_length",
        "-min_length",
        type=int,
        default=0,
        help="Minimum prediction length",
    )
    group.add(
        "--max_length",
        "-max_length",
        type=int,
        default=250,
        help="Maximum prediction length.",
    )
    group.add(
        "--max_length_ratio",
        "-max_length_ratio",
        type=float,
        default=1.25,
        help="Maximum prediction length ratio."
        "for European languages 1.25 is large enough"
        "for target Asian characters need to increase to 2-3"
        "for special languages (burmese, amharic) to 10",
    )
    # Decoding content constraint
    group.add(
        "--block_ngram_repeat",
        "-block_ngram_repeat",
        type=int,
        default=0,
        help="Block repetition of ngrams during decoding.",
    )
    group.add(
        "--ignore_when_blocking",
        "-ignore_when_blocking",
        nargs="+",
        type=str,
        default=[],
        help="Ignore these strings when blocking repeats. "
        "You want to block sentence delimiters.",
    )
    group.add(
        "--replace_unk",
        "-replace_unk",
        action="store_true",
        help="Replace the generated UNK tokens with the "
        "source token that had highest attention weight. If "
        "phrase_table is provided, it will look up the "
        "identified source token and give the corresponding "
        "target token. If it is not provided (or the identified "
        "source token does not exist in the table), then it "
        "will copy the source token.",
    )
    group.add(
        "--ban_unk_token",
        "-ban_unk_token",
        action="store_true",
        help="Prevent unk token generation by setting unk proba to 0",
    )
    group.add(
        "--phrase_table",
        "-phrase_table",
        type=str,
        default="",
        help="If phrase_table is provided (with replace_unk), it will "
        "look up the identified source token and give the "
        "corresponding target token. If it is not provided "
        "(or the identified source token does not exist in "
        "the table), then it will copy the source token.",
    )


def translate_opts(parser, dynamic=False):
    """Translation / inference options"""
    group = parser.add_argument_group("Model")
    group.add(
        "--model",
        "-model",
        dest="models",
        metavar="MODEL",
        nargs="+",
        type=str,
        default=[],
        required=True,
        help="Path to model .pt file(s). "
        "Multiple models can be specified, "
        "for ensemble decoding.",
    )
    group.add(
        "--precision",
        "-precision",
        default="",
        choices=["", "fp32", "fp16", "int8"],
        help="Precision to run inference."
        "default is model.dtype"
        "fp32 to force slow FP16 model on GTX1080"
        "int8 enables pytorch native 8-bit quantization"
        "(cpu only)",
    )
    group.add(
        "--fp32",
        "-fp32",
        action=DeprecateAction,
        help="Deprecated use 'precision' instead",
    )
    group.add(
        "--int8",
        "-int8",
        action=DeprecateAction,
        help="Deprecated use 'precision' instead",
    )
    group.add(
        "--avg_raw_probs",
        "-avg_raw_probs",
        action="store_true",
        help="If this is set, during ensembling scores from "
        "different models will be combined by averaging their "
        "raw probabilities and then taking the log. Otherwise, "
        "the log probabilities will be averaged directly. "
        "Necessary for models whose output layers can assign "
        "zero probability.",
    )

    group = parser.add_argument_group("Data")
    group.add(
        "--data_type",
        "-data_type",
        default="text",
        help="Type of the source input. Options: [text].",
    )

    group.add(
        "--src",
        "-src",
        required=True,
        help="Source sequence to decode (one line per " "sequence)",
    )
    group.add("--tgt", "-tgt", help="True target sequence (optional)")
    group.add(
        "--tgt_file_prefix",
        "-tgt_file_prefix",
        action="store_true",
        help="Generate predictions using provided `-tgt` as prefix.",
    )
    group.add(
        "--output",
        "-output",
        default="pred.txt",
        help="Path to output the predictions (each line will "
        "be the decoded sequence",
    )
    group.add(
        "--report_align",
        "-report_align",
        action="store_true",
        help="Report alignment for each translation.",
    )
    group.add(
        "--gold_align",
        "-gold_align",
        action="store_true",
        help="Report alignment between source and gold target."
        "Useful to test the performance of learnt alignments.",
    )
    group.add(
        "--report_time",
        "-report_time",
        action="store_true",
        help="Report some translation time metrics",
    )
    group.add(
        "--profile",
        "-profile",
        action="store_true",
        help="Report pytorch profiling stats",
    )
    # Adding options related to source and target features
    _add_features_opts(parser)

    # Adding options relate to decoding strategy
    _add_decoding_opts(parser)

    # Adding option for logging
    _add_logging_opts(parser, is_train=False)

    distributed_opts(parser)

    group = parser.add_argument_group("Efficiency")
    group.add("--batch_size", "-batch_size", type=int, default=30, help="Batch size")
    group.add(
        "--batch_type",
        "-batch_type",
        default="sents",
        choices=["sents", "tokens"],
        help="Batch grouping for batch_size. Standard "
        "is sents. Tokens will do dynamic batching",
    )
    group.add("--gpu", "-gpu", type=int, default=-1, help="Device to run on")

    if dynamic:
        group.add(
            "-transforms",
            "--transforms",
            default=[],
            nargs="+",
            choices=AVAILABLE_TRANSFORMS.keys(),
            help="Default transform pipeline to apply to data.",
        )

        # Adding options related to Transforms
        _add_dynamic_transform_opts(parser)

    _add_quant_opts(parser)


# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


class StoreLoggingLevelAction(configargparse.Action):
    """Convert string to logging level"""

    import logging

    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs
        )

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """Deprecate action"""

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(
            option_strings, dest, nargs=0, help=help, **kwargs
        )

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
