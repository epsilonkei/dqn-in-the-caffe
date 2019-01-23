#include <cmath>
#include <ctime>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.cpp"
#include "util/stop_watch.hpp"

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
DEFINE_int32(steps_per_epoch, 5000, "Number of training and evaluating steps per epoch");
DEFINE_int32(max_iter, 10000000, "Network stop training after max_iter number of iterations.");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

#if ENABLE_TIMER
stop_watch timer = stop_watch();
#endif //ENABLE_TIMER

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
#if ENABLE_TIMER
      timer.start();
#endif //ENABLE_TIMER
      const auto action = dqn.SelectAction(input_frames, epsilon);
#if ENABLE_TIMER
      timer.stop();
      std::cerr << "Elapsed time: " << timer.getTime() << std::endl;
      timer.reset();
#endif //ENABLE_TIMER
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
  dqn::DQN dqn(legal_actions, FLAGS_solver, FLAGS_memory, FLAGS_gamma, FLAGS_verbose,
               log_dir + "/q_log.csv", FLAGS_steps_per_epoch);
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

  int eval_epoc = 0, train_epoc_number;
  double epsilon;
  double total_score = 0.0, train_score, epoch_total_score = 0.0, eval_score;
  int epoch_episode_count = 0.0;
  double total_time = 0.0, hours, hours_for_million;
  int next_epoch_boundry = FLAGS_steps_per_epoch;
  double running_average = 0.0, plot_average_discount = 0.05;

  std::ofstream train_log(log_dir + "/train_log.csv");
  std::ofstream eval_log(log_dir + "/eval_log.csv");
  std::ofstream rom_info(log_dir + "/rom_info.txt");
  rom_info << FLAGS_rom << std::endl;
  // eval_log << "Epoch,Evaluate score,Hours training" << std::endl;
  // train_log << "Epoch,Epoch avg score,Hours training,Number of episodes"
  //   ",Episodes in epoch" << std::endl;
  caffe::Timer run_timer;

  for (auto episode = 0;; episode++) {
    if (FLAGS_verbose)
      std::cout << "Episode: " << episode << std::endl;
    run_timer.Start();

    epoch_episode_count++;
    epsilon = CalculateEpsilon(dqn.current_iteration());
    train_score = PlayOneEpisode(ale, dqn, epsilon, true);
    epoch_total_score += train_score;

    if (dqn.current_iteration() > 0)  // started training?
      total_time += run_timer.MilliSeconds();

    if (episode == 0)
      running_average = train_score;
    else
      running_average = train_score*plot_average_discount
        + running_average*(1.0-plot_average_discount);

    if (dqn.current_iteration() >= next_epoch_boundry) {
      hours =  total_time / 1000. / 3600.;
      train_epoc_number = static_cast<int>((next_epoch_boundry)/FLAGS_steps_per_epoch);
      hours_for_million = hours/(dqn.current_iteration()/1000000.0);
      std::cout << "Epoch(" << train_epoc_number
                << ":" << dqn.current_iteration() << "): "
                << "average score " << running_average << " in "
                << hours << " hour(s)" << std::endl;
      std::cout << "Estimated Time for 1 million iterations: "
                << hours_for_million << " hours" << std::endl;

      train_log << train_epoc_number << " " << running_average << " " << hours
        << " " << episode << " " << epoch_episode_count << std::endl;

      epoch_total_score = 0.0;
      epoch_episode_count = 0;

      while (next_epoch_boundry < dqn.current_iteration())
        next_epoch_boundry += FLAGS_steps_per_epoch;

      // Evaluate after every epoch evaluate the current strength
      eval_score = PlayOneEpisode(ale, dqn, 0.05, false);
      std::cout << "Evaluation score: " << eval_score << std::endl;
      eval_log << eval_epoc << " " << eval_score << " " <<
        run_timer.MilliSeconds() / 1000. / 3600. << std::endl;
      eval_epoc ++;
    }

    if (dqn.current_iteration() >= FLAGS_max_iter) break;
  }
  train_log.close();
  eval_log.close();
  dqn.CloseQLog();
};
