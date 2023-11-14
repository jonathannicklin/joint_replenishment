#pragma once
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "dynaplex/system.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/sampledata.h"
#include "dynaplex/sample.h"
#include "dynaplex/uniformactionselector.h"
#include "dynaplex/sequentialhalving.h"
#include "dynaplex/policytrainer.h"
#include <tuple>
namespace DynaPlex::Algorithms {
	class DCL
	{
	public:
		DCL(const DynaPlex::System& system, DynaPlex::MDP mdp, DynaPlex::Policy policy_0=nullptr, const DynaPlex::VarGroup& config = VarGroup{});
		/// Trains a number of policies, each using data generated by the previous generation. 
		void TrainPolicy();
		///  Gets the trained policy of the specified generation. Setting generation to -1 (default) will yield the latest generation.  
		DynaPlex::Policy GetPolicy(int64_t generation = -1);
		/// Gets a vector of all trained policies.
		std::vector<DynaPlex::Policy> GetPolicies();
		
		std::string GenerateFeatsSamples(DynaPlex::Policy policy, int64_t generation);
		std::string GetPathOfFeatsSampleFile(int64_t generation);
		std::tuple<std::string,std::string,std::string> GetPathsOfPolicyFiles(int64_t generation);
	
	private:
		void GenerateSamples(DynaPlex::Policy policy, int64_t generation);
		std::string GetPathOfSampleFile(int64_t generation, int rank = -1);
	
		void GenerateSamplesOnThread(std::span<DynaPlex::NN::Sample>, DynaPlex::Policy, int32_t);
	
		//for a progress count when generating samples accross threads. 
		std::shared_ptr<std::atomic<int64_t>> total_samples_collected;

		int32_t rng_seed;
		int32_t node_sampling_offset;
		int64_t sampling_time_out,H,M,N,num_gens,L,reinitiate_counter,json_save_format;

		bool retrain_lastgen_only, silent;


		DynaPlex::NN::PolicyTrainer trainer;


		DynaPlex::VarGroup nn_architecture = DynaPlex::VarGroup{};

		DynaPlex::MDP mdp;
		DynaPlex::Policy policy_0;
		DynaPlex::System system;
		DynaPlex::DCL::UniformActionSelector uniform_action_selector;
		DynaPlex::DCL::SequentialHalving sequentialhalving_action_selector;

	};
}