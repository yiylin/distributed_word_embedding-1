#include <thread>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <new>

#include "dictionary.h"
#include "huffman_encoder.h"
#include "util.h"
#include "reader.h"
#include "multiverso.h"
#include "barrier.h"
#include "Distributed_wordembedding.h"
#include "parameter_loader.h"
#include "trainer.h"
#include "word_embedding.h"
#include "memory_manager.h"

using namespace multiverso;
using namespace wordembedding;

bool ReadWord(char *word, FILE *fin)
{
	int idx = 0;
	char ch;
	while (!feof(fin))
	{
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
		{
			if (idx > 0)
			{
				if (ch == '\n')
					ungetc(ch, fin);
				break;
			}

			if (ch == '\n')
			{
				strcpy(word, (char *)"</s>");
				return true;
			}
			else
			{
				continue;
			}
		}

		word[idx++] = ch;
		if (idx >= kMaxString - 1)
			idx--;
	}

	word[idx] = 0;
	return idx > 0;
}


//Read the vocabulary file; create the dictionary
//and huffman_encoder according opt
int64 LoadVocab(Option *opt,
	Dictionary *dictionary, HuffmanEncoder *huffman_encoder)
{
	int64 total_words = 0;
	char word[kMaxString];
	FILE* fid = nullptr;
	if (opt->read_vocab_file != nullptr && strlen(opt->read_vocab_file) > 0)
	{
		fid = fopen(opt->read_vocab_file, "r");
		int word_freq;
		while (fscanf(fid, "%s %d", word, &word_freq) != EOF)
		{
			dictionary->Insert(word, word_freq);
		}
	}

	dictionary->RemoveWordsLessThan(opt->min_count);
	//multiverso::Log::Info("Dictionary size: %d\n", dictionary->Size());
	total_words = 0;
	for (int i = 0; i < dictionary->Size(); ++i)
		total_words += dictionary->GetWordInfo(i)->freq;
	//multiverso::Log::Info("Words in Dictionary %I64d\n", total_words);
	if (opt->hs)
		huffman_encoder->BuildFromTermFrequency(dictionary);
	if (fid != nullptr)
		fclose(fid);

	return total_words;
}

real learning_rate;
int64 word_count_actual;

/*!
* \brief Create memory for weight_IE_ weight_EO_ sum_gradient2_IE_ sum_gradient2_EO_
*/

/*!
* \brief TrainNN
* \param data_block represents the trainNNing datablock
* \param index_start the thread's starting index in the sentence vector
* \param interval the total_number of thread
* \param word_count count the words which has been processed by trainNN
* \param hidden_act  hidden layer value
* \param hidden_err  hidden layer error
*/
void Train(DataBlock *data_block, int index_start,
	int interval, int64& word_count,
	real* hidden_act, real* hidden_err, Reader* reader);
/*!
* \brief PrepareParameter for parameterloader threat
* \param data_block datablock for parameterloader to parse
* \param input_nodes  input_nodes represent the parameter which input_layer includes
* \param output_nodes output_nodes represent the parameter which output_layer inclueds

/*!
* \brief Update the learning rate
*/
void UpdateLearningRate();


			Option *option_;
			Dictionary *dictionary_;
			HuffmanEncoder *huffmanEncoder_;
			Sampler *sampler_;
			std::unordered_set<int> input_nodes_, output_nodes_;
			int dictionary_size_;
			real *weight_IE_;
			real *weight_EO_;
			real*sum_gradient2_IE_;
			real* sum_gradient2_EO_;
			int feat[kMaxSentenceLength + 1];

			void MallocMemory()
			{
				weight_IE_ = new (std::nothrow)real[(long long)dictionary_size_ * option_->embeding_size];
				// assert(weight_IE_ != nullptr);
				weight_EO_ = new (std::nothrow)real[(long long)dictionary_size_ * option_->embeding_size];
			}
			/*!
			* \brief Parse the needed parameter in a window
			*/

			void TrainParse(int *feat, int feat_cnt, int word_idx, real *hidden_act, real *hidden_err);
			/*!
			* \brief Parse a sentence and deepen into two branchs
			* \one for TrainNN,the other one is for Parameter_parse&request
			*/

			/*!
			* \brief Get the hidden layer vector
			* \param input_nodes represent the input nodes
			* \param hidden_act store the hidden layer vector

			void FeedForward(int* feat, int feat_cnt, real* hidden_act);
			/*!
			* \brief Calculate the hidden_err and update the output-embedding weight
			* \param label record the label of every output-embedding vector
			* \param word_idx the index of the output-embedding vector
			* \param classifier store the output-embedding vector
			* \param store the hidden layer vector
			* \param store the hidden-error which is used
			* \to update the input-embedding vector
			*/
			void BPOutputLayer(int label, real* classifier,
				real* hidden_act, real* hidden_err);
			/*!
			* \brief Copy the input_nodes&output_nodes to WordEmbedding private set
			*/

			/*!
			* \brief Train a window sample and update the
			* \input-embedding&output-embedding vectors
			* \param input_nodes represent the input nodes
			* \param output_nodes represent the ouput nodes
			* \param hidden_act  store the hidden layer vector
			* \param hidden_err  store the hidden layer error
			*/

			/*!
			* \brief Train the sentence actually
			*/
			void Train(int* sentence, int sentence_length,
				uint64 next_random, real* hidden_act, real* hidden_err);

		
			//Train neural networks of WordEmbedding
			void Train(DataBlock *data_block, int index_start, int interval,
				int64& word_count, real* hidden_act, real* hidden_err, Reader* reader)
			{
				int sentence[kMaxSentenceLength + 2] = { 0 };

				printf("WordEmbedding Train\n");
				int64 last_word_count = word_count;
				clock_t start = clock();
				int64 total = 0;

				while (1)
				{
					int64 word_count_deta;
					uint64 next_random = (uint64)rand() * 10000 + (uint64)rand();
					//printf("sentence_leng\n");
					int sentence_length = reader->GetSentence(sentence, word_count_deta);
					//data_block->GetSentence(i, sentence, sentence_length,
					//    word_count_deta, next_random);
					//printf("sentence_length = %d\n", sentence_length);
					if (sentence_length == 0)
						break;

					for (int sentence_position = 0; sentence_position < sentence_length; ++sentence_position)
					{
						if (sentence[sentence_position] == -1) continue;
						int off = rand() % 5;
						int feat_size = 0;
						for (int i = off; i < 5 * 2 + 1 - off; ++i)
						if (i != 5)
						{
							int c = sentence_position - 5 + i;
							if (c < 0 || c >= sentence_length || sentence[c] == -1)
								continue;

							feat[feat_size++] = sentence[c];
						}

						if (option_->cbow) 	//train cbow
						{
							TrainParse(feat, feat_size, sentence[sentence_position],
								hidden_act, hidden_err);
						}
					}

					word_count += word_count_deta;
					if (word_count > last_word_count + 10000)
					{
						total += word_count - last_word_count;
						printf("TrainNNSpeed: Words/thread/second %lfk\n",
							((double)total) /
							(clock() - start) * (double)CLOCKS_PER_SEC / 1000);
						last_word_count = word_count;

					}
				}
			}
			//Update the learning rate
			void UpdateLearningRate()
			{
				if (option_->use_adagrad == false)
				{
					learning_rate = static_cast<real>(option_->init_learning_rate *
						(1 - word_count_actual / ((real)option_->total_words * option_->epoch + 1.0)));
					if (learning_rate < option_->init_learning_rate * 0.0001)
						learning_rate = static_cast<real>(option_->init_learning_rate * 0.0001);
				}
			}

	
			inline void FeedForward(int* feat, int feat_cnt, real* hidden_act)
			{
				for (int i = 0; i < feat_cnt; ++i)
				{
					real* input_embedding = weight_IE_ + feat[i] * option_->embeding_size;
					for (int j = 0; j < option_->embeding_size; ++j)
						hidden_act[j] += input_embedding[j];
				}

				//Change2 .............................................
				if (feat_cnt > 1)
				{
					for (int j = 0; j < option_->embeding_size; ++j)
						hidden_act[j] /= feat_cnt;
				}
			}

			inline void TrainParse(int *feat, int feat_cnt, int word_idx, real *hidden_act, real *hidden_err)
			{
				memset(hidden_act, 0, option_->embeding_size * sizeof(real));
				memset(hidden_err, 0, option_->embeding_size * sizeof(real));
				FeedForward(feat, feat_cnt, hidden_act);

				if (option_->hs)
				{
					auto info = huffmanEncoder_->GetLabelInfo(word_idx);
					for (int d = 0; d < info->codelen; d++)
						BPOutputLayer(info->code[d], weight_EO_ + info->point[d] * option_->embeding_size,
						hidden_act, hidden_err);
				}
				else
				if (option_->negative_num)
				{
					BPOutputLayer(1, weight_EO_ + word_idx* option_->embeding_size,
						hidden_act, hidden_err);
					for (int d = 0; d < option_->negative_num; d++)
					{
						int target = sampler_->NegativeSampling();
						if (target == word_idx) continue;
						BPOutputLayer(0, weight_EO_ + target  * option_->embeding_size,
							hidden_act, hidden_err);
					}
				}

				
					for (int i = 0; i < feat_cnt; ++i)
					{
						int &node_id = feat[i];
						real* input_embedding = weight_IE_ + node_id  * option_->embeding_size;
						//assert(input_embedding != nullptr);
						for (int j = 0; j < option_->embeding_size; ++j)
							input_embedding[j] += hidden_err[j];
					}
				
			}


			//Train with inverse direction and update the hidden-output 
			inline void BPOutputLayer(int label,
				real* classifier, real* hidden_act, real* hidden_err)
			{
				//assert(classifier != nullptr && hidden_act != nullptr && hidden_err != nullptr);
				real f = 0;
				//Propagate hidden -> output
				for (int j = 0; j < option_->embeding_size; ++j)
					f += hidden_act[j] * classifier[j];
				f = 1 / (1 + exp(-f));
				real error = (1 - label - f) * learning_rate;
				//Propagate errors output -> hidden
				for (int j = 0; j < option_->embeding_size; ++j)
					hidden_err[j] += error * classifier[j];

					for (int j = 0; j < option_->embeding_size; ++j)
						classifier[j] += error * hidden_act[j];
				
			}

int main(int argc, char *argv[])
{   
    try
    {
        //Distributed_wordembedding *ptr = new (std::nothrow)Distributed_wordembedding();
       // assert(ptr != nullptr);
        //ptr->Run(argc, argv);

		//The barrier for trainers
		g_log_suffix = GetSystemTime();
		srand(static_cast<unsigned int>(time(NULL)));
		option_ = new (std::nothrow)Option();
		assert(option_ != nullptr);
		dictionary_ = new (std::nothrow)Dictionary();
		assert(dictionary_ != nullptr);
		huffmanEncoder_ = new (std::nothrow)HuffmanEncoder();
		
		//Parse argument and store them in option

		if (argc <= 1)
		{
			option_->PrintUsage();
			return 0;
		}

		option_->ParseArgs(argc, argv);
		//Read the vocabulary file; create the dictionary
		//and huffman_encoder according opt

		//multiverso::Log::Info("Loading vocabulary ...\n");
		option_->total_words = LoadVocab(option_, dictionary_,
			huffmanEncoder_);
		//multiverso::Log::Info("Loaded vocabulary\n");

		option_->PrintArgs();

		
		if (option_->negative_num)
			Sampler::SetNegativeSamplingDistribution(dictionary_);

		char *filename = new (std::nothrow)char[strlen(option_->train_file) + 1];
		assert(filename != nullptr);
		strcpy(filename, option_->train_file);
		Reader *reader_ = new (std::nothrow)Reader(dictionary_, option_, sampler_, filename);
		assert(reader_ != nullptr);

		
		dictionary_size_ = dictionary_->Size();
		//Step 1, Create Multiverso ParameterLoader and Trainers, 
		//Start Multiverso environment
		MallocMemory();
		
		int data_block_count = 0;
		//int64 file_size = GetFileSize(option_->train_file);
		//multiverso::Log::Info("train-file-size:%lld, data_block_size:%lld\n",
		//	file_size, option_->data_block_size);
		//start_ = clock();
		// multiverso::Multiverso::BeginTrain();
		int64 word_count = 0;
		real *hidden_act_ = new real[option_->embeding_size], *hidden_err_ = new real[option_->embeding_size];
		for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch)
		{
			reader_->ResetStart();
			//multiverso::Multiverso::BeginClock();
			//for (int64 cur = 0; cur < file_size; cur += option_->data_block_size)
			Train(nullptr, 0, option_->thread_cnt,
				word_count, hidden_act_, hidden_err_, reader_);
			//multiverso::Multiverso::EndClock();
		}

		delete hidden_act_;
		delete hidden_err_;
    }
    catch (std::bad_alloc &memExp)
    {
        multiverso::Log::Info("Something wrong with new() %s\n", memExp.what());
    }
    catch(...)
    {
        multiverso::Log::Info("Something wrong with other reason!\n");
    }
    return 0;
}
