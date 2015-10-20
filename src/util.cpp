#include "util.h"

namespace multiverso
{
    namespace wordembedding
    {
        Option::Option()
        {
            train_file = NULL;
            read_vocab_file = NULL;
            output_file = NULL;
            sw_file = NULL;
            endpoints_file = "";
            hs = true;
            negative_num = 0;
            output_binary = false;
            sample = 0;
            cbow = true;
            embeding_size = 0;
            thread_cnt = 1;
            window_size = 5;
            min_count = 5;
            data_block_size = 1000000;
            init_learning_rate = static_cast<real>(0.025);
            epoch = 1;
            stopwords = false;
            is_pipeline = true;
            total_words = 0;
            max_preload_data_size = 8000000000LL;
            use_adagrad = false;
            //multiverso config
            num_servers = 0;
            num_aggregator = 1;
            lock_option = 1;
            num_lock = 100;
            max_delay = 0;
        }
        //Input all the local model-arguments 
        void Option::ParseArgs(int argc, char* argv[])
        {
            for (int i = 1; i < argc; i += 2)
            {
                if (strcmp(argv[i], "-size") == 0) embeding_size = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-train_file") == 0) train_file = argv[i + 1];
                if (strcmp(argv[i], "-endpoints_file") == 0) endpoints_file = argv[i + 1];
                if (strcmp(argv[i], "-read_vocab") == 0) read_vocab_file = argv[i + 1];
                if (strcmp(argv[i], "-binary") == 0) output_binary = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-cbow") == 0) cbow = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-alpha") == 0) init_learning_rate = static_cast<real>(atof(argv[i + 1]));
                if (strcmp(argv[i], "-output") == 0) output_file = argv[i + 1];
                if (strcmp(argv[i], "-window") == 0) window_size = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-sample") == 0) sample = static_cast<real>(atof(argv[i + 1]));
                if (strcmp(argv[i], "-hs") == 0) hs = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-data_block_size") == 0) data_block_size = atoll(argv[i + 1]);
                if (strcmp(argv[i], "-max_preload_data_size") == 0) max_preload_data_size = atoll(argv[i + 1]);
                if (strcmp(argv[i], "-negative") == 0) negative_num = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-threads") == 0) thread_cnt = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-min_count") == 0) min_count = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-epoch") == 0) epoch = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-stopwords") == 0) stopwords = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-sw_file") == 0)  sw_file = argv[i + 1];
                if (strcmp(argv[i], "-use_adagrad") == 0) use_adagrad = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-is_pipeline") == 0) is_pipeline = (atoi(argv[i + 1]) != 0);
                if (strcmp(argv[i], "-num_servers") == 0) num_servers = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-num_aggregator") == 0) num_aggregator = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-lock_option") == 0) lock_option = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-num_lock") == 0) num_lock = atoi(argv[i + 1]);
                if (strcmp(argv[i], "-max_delay") == 0) max_delay = atoi(argv[i + 1]);
				
            }
        }

        void Option::PrintUsage()
        {
            puts("Usage:");
            puts("-size: word embedding size, e.g. 300");
            puts("-train_file: the training corpus file, e.g.enwik2014");
            puts("-read_vocab : the file to read all the vocab counts info");
            puts("-binary : 0 or 1, indicates whether to write all the embeddings vectors into binary format");
            puts("-cbow : 0 or 1, default 1, whether to use cbow or not");
            puts("-alpha : initial learning rate, usually set to 0.025");
            puts("-output : the output file to store all the embedding vectors");
            puts("-window : the window size");
            puts("-sample : the sub - sample size, usually set to 0");
            puts("-hs : 0 or 1, default 1, whether to use hierarchical softmax");
            puts("-negative : the negative word count in negative sampling, please set it to 0 when - hs = 1");
            puts("-threads : the thread number to run in one machine");
            puts("-min_count : words with lower frequency than min_count is removed from dictionary");
            puts("-epoch : the epoch number");
            puts("-stopwords : 0 or 1, whether to avoid training stop words");
            puts("-sw_file : the stop words file storing all the stop words, valid when -stopwords = 1");
            puts("-use_adagrad : 0 or 1, whether to use adagrad to adjust learnin rate");
            puts("-data_block_size : default 1MB, the maximum bytes which a data block will store");
            puts("-max_preload_data_size : default 8GB, the maximum data size(bytes) which multiverse_WordEmbedding will preload");
            puts("-num_servers : default 0, the parameter of multiverso.Separately, 0 indicates all precesses are servers");
            puts("-num_aggregator : default 1, number of aggregation threads in a process");
            puts("-max_delay : default 0, the delay bound(max staleness)");
            puts("-num_lock : default 100, number of locks in Locked option");
            puts("-is_pipeline : 0 or 1, whether to use pipeline");
            puts("-lock_option : default 0, Lock option. 0 : the trheads do not write and there is no contention; 1:there is no lock for thread contention; 2:normal lock for thread contention");
            puts("-server_endpoint_file : default "", server ZMQ socket endpoint file in MPI - free version");
        }

        void Option::PrintArgs()
        {
            multiverso::Log::Info("train_file: %s\n", train_file);
            multiverso::Log::Info("read_vocab_file: %s\n", read_vocab_file);
            multiverso::Log::Info("output_file: %s\n", output_file);
            multiverso::Log::Info("sw_file: %s\n", sw_file);
            multiverso::Log::Info("hs: %d\n", hs);
            multiverso::Log::Info("output_binary: %d\n", output_binary);
            multiverso::Log::Info("cbow: %d\n", cbow);
            multiverso::Log::Info("stopwords: %d\n", stopwords);
            multiverso::Log::Info("use_adagrad: %d\n", use_adagrad);    
            multiverso::Log::Info("sample: %lf\n", sample);
            multiverso::Log::Info("embeding_size: %d\n", embeding_size);
            multiverso::Log::Info("thread_cnt: %d\n", thread_cnt);
            multiverso::Log::Info("window_size: %d\n", window_size);
            multiverso::Log::Info("negative_num: %d\n", negative_num);
            multiverso::Log::Info("min_count: %d\n", min_count);
            multiverso::Log::Info("epoch: %d\n", epoch);
            multiverso::Log::Info("total_words: %lld\n", total_words);
            multiverso::Log::Info("max_preload_data_size: %lld\n", max_preload_data_size);
            multiverso::Log::Info("init_learning_rate: %lf\n", init_learning_rate);
            multiverso::Log::Info("data_block_size: %lld\n", data_block_size);
            multiverso::Log::Info("num_servers: %d\n", num_servers);
            multiverso::Log::Info("num_aggregator: %d\n", num_aggregator);
            multiverso::Log::Info("is_pipeline: %d\n", is_pipeline);
            multiverso::Log::Info("lock_option: %d\n", lock_option);
            multiverso::Log::Info("num_lock: %d\n", num_lock);
            multiverso::Log::Info("max_delay: %d\n", max_delay);
            multiverso::Log::Info("endpoints_file: %s\n", endpoints_file);
        }

		int* Sampler::table_ = NULL;
		std::default_random_engine Sampler::generator;
		std::uniform_int_distribution<int> Sampler::int_distribution(0, kTableSize - 1);
        
		std::string GetSystemTime()
        {
            time_t t = time(0);
            char tmp[128];
            strftime(tmp, sizeof(tmp), "%Y%m%d%H%M%S", localtime(&t));
            return std::string(tmp);
        }

        std::string g_log_suffix;   
    }
}