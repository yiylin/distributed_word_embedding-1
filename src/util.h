#pragma once

/*!
* file util.h
* \brief Struct Option stores many general arguments in model
*/
#define NDEBUG

#include <cstring>
#include <cstdlib>
#include <random>
#include <cassert>
#include <exception>

#include "constant.h"
#include "dictionary.h"

namespace multiverso
{
    namespace wordembedding
    {
        struct Option
        {
            const char* train_file;
            const char* read_vocab_file;
            const char* output_file;
            const char* sw_file;
            const char* endpoints_file;
            bool hs, output_binary, cbow, stopwords;
            bool use_adagrad;
            bool is_pipeline;
            real sample;
            int64 data_block_size;
            int embeding_size, thread_cnt, window_size, negative_num, min_count, epoch; 
            int64 total_words;
            int64 max_preload_data_size;
            real init_learning_rate;
            int num_servers, num_aggregator, lock_option, num_lock, max_delay;

            Option();
            /*!
            * \brief Get the model-set arguments from file  
            */
            void ParseArgs(int argc, char* argv[]);
            void PrintArgs();
            void PrintUsage();

        };


        class Sampler
        {
        public:
            Sampler();
            /*!
            * \brief Set the negative-sampling distribution for every vocabulary
            * \param dictionary the train_file dictionary
            */
            void SetNegativeSamplingDistribution(Dictionary *dictionary);
            bool WordSampling(int64 word_cnt, int64 train_words, real sample);
            /*!
            * \brief Get the next random according to the existing random seed
            */
            uint64 GetNextRandom(uint64 next_random);
            int NegativeSampling(uint64 next_random);

        private:
            int* table_;

            //No copying allowed
            Sampler(const Sampler&);
            void operator=(const Sampler&);
        };

        std::string GetSystemTime();
        extern std::string g_log_suffix;
    }
}