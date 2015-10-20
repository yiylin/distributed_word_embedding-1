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
           
            /*!
            * \brief Set the negative-sampling distribution for every vocabulary
            * \param dictionary the train_file dictionary
            */
			//Set the negative-sampling distribution
			static void Sampler::SetNegativeSamplingDistribution(Dictionary *dictionary)
			{
				real train_words_pow = 0;
				real power = 0.75;
				table_ = (int *)malloc(kTableSize * sizeof(int));
				for (int i = 0; i < dictionary->Size(); ++i)
					train_words_pow += static_cast<real>(pow(dictionary->GetWordInfo(i)->freq, power));
				int cur_pos = 0;
				real d1 = (real)pow(dictionary->GetWordInfo(cur_pos)->freq, power)
					/ (real)train_words_pow;

				assert(table_ != nullptr);
				for (int i = 0; i < kTableSize; ++i)
				{
					table_[i] = cur_pos;
					if (i > d1 * kTableSize && cur_pos + 1 < dictionary->Size())
					{
						cur_pos++;
						d1 += (real)pow(dictionary->GetWordInfo(cur_pos)->freq, power)
							/ (real)train_words_pow;
					}
				}
			}

			static bool Sampler::WordSampling(int64 word_cnt,
				int64 train_words, real sample)
			{
				real ran = (sqrt(word_cnt / (sample * train_words)) + 1) *
					(sample * train_words) / word_cnt;
				return (ran > ((real)rand() / (RAND_MAX)));
			}
			//Get the next random 
			static uint64 Sampler::GetNextRandom(uint64 next_random)
			{
				return next_random * (uint64)25214903917 + 11;
			}

			static int Sampler::NegativeSampling()
			{
				return table_[(int_distribution)(generator)];
			}


        private:
            static int* table_;
		    static std::default_random_engine generator;
			static std::uniform_int_distribution<int> int_distribution;

            //No copying allowed
            Sampler(const Sampler&);
            void operator=(const Sampler&);
        };

		std::string GetSystemTime();
        extern std::string g_log_suffix;
    }
}