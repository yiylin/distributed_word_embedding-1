#include "trainer.h"
namespace multiverso
{
    namespace wordembedding
    {
        Trainer::Trainer(int trainer_id, Option *option,
            multiverso::Barrier *barrier,
            Dictionary* dictionary, WordEmbedding* WordEmbedding,
            MemoryManager* memory_mamanger, Reader* reader)
        {
			reader_ = reader;
            trainer_id_ = trainer_id;
            option_ = option;   
            word_count = 0;
            WordEmbedding_ = WordEmbedding;
            barrier_ = barrier;
            dictionary_ = dictionary;
            memory_mamanger_ = memory_mamanger;
            hidden_act_ = (real *)calloc(option_->embeding_size, sizeof(real));
            hidden_err_ = (real *)calloc(option_->embeding_size, sizeof(real));
            process_count_ = -1;
            process_id_ = -1;   
            assert(hidden_act_ != nullptr);
            assert(hidden_err_ != nullptr);
            start_ = 0;
            train_count_ = 0;
            if (trainer_id_ == 0)
            {
                //The log which recordes the begin and end time of TrainIteration()
                char log_name[100];
                sprintf(log_name, "trainer%s.txt", g_log_suffix.c_str());
                log_file_ = fopen(log_name, "w");
            }
        }


        void Trainer::TrainIteration(multiverso::DataBlockBase *data_block)
        {
            if (process_id_ == -1)
                process_id_ = multiverso::Multiverso::ProcessRank();

            if (trainer_id_ == 0)
                //Record the starting time of the Trainiteration  
                fprintf(log_file_, "%lf\n", (clock()) / (double)CLOCKS_PER_SEC);

            multiverso::Log::Info("Rank %d Train %d Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            ++train_count_;
            //Compute the total number of processes
            if (process_count_ == -1)
                process_count_ = multiverso::Multiverso::TotalProcessCount();
        
			DataBlock *data = static_cast<DataBlock*>(data_block);
            //Step 2, After finishing copying parameter,
            //Use WordEmbedding_ to train a part of data_block
            int64 last_word_count = word_count;
            clock_t start = clock();
            multiverso::Log::Debug("Rank %d Train %d TrainNN Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            WordEmbedding_->Train(data, trainer_id_, option_->thread_cnt,
                word_count, hidden_act_, hidden_err_, reader_);
            if (word_count > last_word_count)
            {
                multiverso::Log::Info("TrainNNSpeed: Words/thread/second %lfk\n",
                    ((double)word_count - last_word_count) / 
                    (clock() - start) * (double)CLOCKS_PER_SEC / 1000);
            }
        }

        void Trainer::CopyRow(real* ptr, multiverso::Row<real>& row, int size)
        {
            for (int i = 0; i < size; ++i)
                ptr[i] = row.At(i);
        }


        void Trainer::CopyParameter(std::vector<int>& input_nodes,
            std::vector<int>& output_nodes)
        {
            //Compute the number of necessary memory blocks to store parameter
            std::vector<real*> blocks;
            int current_block = 0;
            size_t total_blocks = (input_nodes.size() + output_nodes.size());
            if (option_->use_adagrad)
                total_blocks *= 2;

            //Request blocks to store parameters
            memory_mamanger_->RequestBlocks(total_blocks, blocks);
            assert(blocks.size() == total_blocks);
            if (blocks.size() != total_blocks)
            {
                multiverso::Log::Error("Rank %d Trainer %d Error to requestBlocks to CopyParameter, allocated_blocks_num=%lld, needed_blocks_num=%lld\n",
                    multiverso::Multiverso::ProcessRank(), trainer_id_, blocks.size(), total_blocks);
                return;
            }

            //Copy input-embedding weights from multiverso to WordEmbedding
            for (int i = 0; i < input_nodes.size(); ++i)
            {
                real* ptr = blocks[current_block++];
                assert(ptr != nullptr);
                CopyRow(ptr, GetRow<real>(kInputEmbeddingTableId,
                    input_nodes[i]), option_->embeding_size);

                //WordEmbedding_->SetWeightIE(input_nodes[i], ptr);
            }

            //Copy embedding-output weights from multiverso to WordEmbedding
            for (int i = 0; i < output_nodes.size(); ++i)
            {
                real* ptr = blocks[current_block++];
                assert(ptr != nullptr);
                CopyRow(ptr, GetRow<real>(kEmbeddingOutputTableId,
                    output_nodes[i]), option_->embeding_size);

                //WordEmbedding_->SetWeightEO(output_nodes[i], ptr);
            }

            if (option_->use_adagrad)
            {
                //Copy input-embedding sum of squarsh of gradient 
                for (int i = 0; i < input_nodes.size(); ++i)
                {
                    real* ptr = blocks[current_block++];
                    assert(ptr != nullptr);
                    CopyRow(ptr, GetRow<real>(kSumGradient2IETableId,
                        input_nodes[i]), option_->embeding_size);

                    //WordEmbedding_->SetSumGradient2IE(input_nodes[i], ptr);
                }

                //Copy embedding-output sum of squarsh of gradient 
                for (int i = 0; i < output_nodes.size(); ++i)
                {
                    real* ptr = blocks[current_block++];
                    assert(ptr != nullptr);
                    CopyRow(ptr, GetRow<real>(kSumGradient2EOTableId,
                        output_nodes[i]), option_->embeding_size);

                    //WordEmbedding_->SetSumGradient2EO(output_nodes[i], ptr);
                }
            }
        }


        void Trainer::AddRow(real* ptr, int table_id, int row_id, int size)
        {
            multiverso::Row<real>& row = GetRow<real>(table_id, row_id);
            for (int i = 0; i < size; ++i)
            {
                real delta = (ptr[i] - row.At(i)) / process_count_;
                if (fabs(delta) > kEps)
                    Add<real>(table_id, row_id, i, delta);
            }
        }

        //Add delta to local buffer and send it to the parameter sever
        void Trainer::AddDeltaParameter(std::vector<int>& input_nodes,
            std::vector<int>& output_nodes)
        {
            std::vector<real*> blocks;
            for (int i = 0; i < input_nodes.size(); ++i)
            {
				real* ptr = nullptr;// WordEmbedding_->GetWeightIE(input_nodes[i]);
                assert(ptr != nullptr);
                AddRow(ptr, kInputEmbeddingTableId, input_nodes[i],
                    option_->embeding_size);

                blocks.push_back(ptr);
            }

            for (int i = 0; i < output_nodes.size(); ++i)
            {
				real* ptr = nullptr;// WordEmbedding_->GetWeightEO(output_nodes[i]);
                assert(ptr != nullptr);
                AddRow(ptr, kEmbeddingOutputTableId, output_nodes[i],
                    option_->embeding_size);
                blocks.push_back(ptr);
            }

            if (option_->use_adagrad)
            {
                for (int i = 0; i < input_nodes.size(); ++i)
                {
					real* ptr = nullptr;// WordEmbedding_->GetSumGradient2IE(input_nodes[i]);
                    assert(ptr != nullptr);
                    AddRow(ptr, kSumGradient2IETableId, input_nodes[i],
                        option_->embeding_size);
                    blocks.push_back(ptr);
                }

                for (int i = 0; i < output_nodes.size(); ++i)
                {
					real* ptr = nullptr;// WordEmbedding_->GetSumGradient2EO(output_nodes[i]);
                    assert(ptr != nullptr);
                    AddRow(ptr, kSumGradient2EOTableId, output_nodes[i],
                        option_->embeding_size);
                    blocks.push_back(ptr);
                }
            }

            //Return all the memory blocks
            memory_mamanger_->ReturnBlocks(blocks);
        }


        void Trainer::SaveEmbedding(const char *file_path, bool is_binary)
        {
            FILE* fid = nullptr;
            if (is_binary)
            {
                fid = fopen(file_path, "wb");
                fprintf(fid, "%d %d\n", dictionary_->Size(),option_->embeding_size);
                for (int i = 0; i < dictionary_->Size(); ++i)
                {
                    fprintf(fid, "%s ",
                        dictionary_->GetWordInfo(i)->word.c_str());

                    multiverso::Row<real>& embedding = GetRow<real>(
                        kInputEmbeddingTableId, i);

                    for (int j = 0; j < option_->embeding_size; ++j)
                    {
                        real tmp = embedding.At(j);
                        fwrite(&tmp, sizeof(real), 1, fid);
                    }

                    fprintf(fid, "\n");
                }

                fclose(fid);
            }
            else
            {
                fid = fopen(file_path, "wt");
                fprintf(fid, "%d %d\n", dictionary_->Size(), option_->embeding_size);
                for (int i = 0; i < dictionary_->Size(); ++i)
                {
                    fprintf(fid, "%s ", dictionary_->GetWordInfo(i)->word.c_str());
                    multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, i);

                    for (int j = 0; j < option_->embeding_size; ++j)
                        fprintf(fid, "%lf ", embedding.At(j));

                    fprintf(fid, "\n");
                }

                fclose(fid);
            }
        }
    }
}