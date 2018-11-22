#include <cmath>
#include <ctime>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <sys/stat.h>
#include "prettyprint.hpp"
#include "dqn.hpp"

DEFINE_bool(verbose, false, "Verbose Output for each frame");
DEFINE_bool(gpu, false, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(rom, "breakout.bin", "Atari 2600 ROM to play");
DEFINE_string(solver, "caffe_mods/dqn_solver.prototxt", "Solver parameter file (*.prototxt)");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory_threshold, 100, "Enough amount of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_string(model, "", "Model file to load");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_double(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_int32(eval_epis_per_epoch, 5000, "Number of episodes per epoch for evaluating while training");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(ALEInterface& ale,
                      dqn::DQN& dqn,
                      const double epsilon,
                      const bool update) {
  assert(!ale.game_over());
  std::deque<dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    if (FLAGS_verbose) {
      std::cout << "frame: " << frame << std::endl;
    }
    const auto current_frame = dqn::PreprocessScreen(ale.getScreen());
    if (FLAGS_show_frame) {
      std::cout << dqn::DrawFrame(*current_frame) << std::endl;
    }
    past_frames.push_back(current_frame);
    if (past_frames.size() < dqn::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
    } else {
      if (past_frames.size() > dqn::kInputFrameCount) {
        past_frames.pop_front();
      }
      dqn::InputFrames input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      const auto action = dqn.SelectAction(input_frames, epsilon);
      auto immediate_score = 0.0;
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        // Last action is repeated on skipped frames
        immediate_score += ale.act(action);
      }
      total_score += immediate_score;
      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      const auto reward =
        immediate_score == 0 ?
        0 :
        immediate_score /= std::abs(immediate_score);
      if (update) {
        // Add the current transition to replay memory
        const auto transition = ale.game_over() ?
          dqn::Transition(input_frames, action, reward, boost::none) :
          dqn::Transition(input_frames,
                          action,
                          reward,
                          dqn::PreprocessScreen(ale.getScreen()));
        dqn.AddTransition(transition);
        // If the size of replay memory is enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.Update();
        }
      }
    }
  }
  ale.reset_game();
  return total_score;
}

void CreateDir(const char* directory) {
  struct stat sb;
  if (!stat(directory, &sb) == 0 || !S_ISDIR(sb.st_mode)) {
    const int dir_err = mkdir(directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if (dir_err == -1) {
         std::cerr <<"Error creating log directory!" << std::endl;
         exit(1);
      }
  }
}

std::string TimeString() {
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[20];
  time (&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer,sizeof(buffer),"%Y%m%d_%H%M%S",timeinfo);
  std::string str(buffer);
  return str;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale(FLAGS_gui);

  // Load the ROM file
  ale.loadROM(FLAGS_rom);

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  std::string log_dir = ".//save_model/" + TimeString();
  CreateDir(log_dir.c_str());
  dqn::DQN dqn(legal_actions, FLAGS_solver, FLAGS_memory, FLAGS_gamma, FLAGS_verbose);
  dqn.Initialize(log_dir);

  if (!FLAGS_model.empty()) {
    // Just evaluate the given trained model
    std::cout << "Loading " << FLAGS_model << std::endl;
  }

  if (FLAGS_evaluate) {
    dqn.LoadTrainedModel(FLAGS_model);
    auto total_score = 0.0;
    for (auto i = 0; i < FLAGS_repeat_games; ++i) {
      std::cout << "Game: " << i << std::endl;
      const auto score =
        PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false);
      std::cout << "Score: " << score << std::endl;
      total_score += score;
    }
    std::cout << "Total_score: " << total_score << std::endl;
    return 0;
  }

  int epoch = 0;
  std::ofstream training_data(log_dir + "/training_log.csv");
  std::ofstream rom_info(log_dir + "/rom_info.txt");
  rom_info << FLAGS_rom << std::endl;
  training_data << "Epoch,Evaluate score,Hours training" << std::endl;
  caffe::Timer run_timer;

  for (auto episode = 0; episode <1 ; episode++) {
    if (FLAGS_verbose)
      std::cout << "Episode: " << episode << std::endl;
    run_timer.Start();

    const auto epsilon = CalculateEpsilon(dqn.current_iteration());
    PlayOneEpisode(ale, dqn, epsilon, true);

    // Evaluate after every 10 episodes evaluate the current strength
    if (dqn.current_iteration() % 10 == 0) {
      const auto eval_score = PlayOneEpisode(ale, dqn, 0.05, false);
      std::cout << "Evaluation score: " << eval_score << std::endl;
      //  Output to log after every N episodes
      if (dqn.current_iteration() % FLAGS_eval_epis_per_epoch == 0) {
        training_data << epoch << ", " << eval_score << ", " <<
          run_timer.MilliSeconds() / 1000. / 3600. << std::endl;
        epoch++;
      }
    }
  }
  training_data.close();
};
