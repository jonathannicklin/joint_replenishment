#pragma once
#include <cstdint>
// #include "mdp.h"
#include "dynaplex/models/joint_replenishment/mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace joint_replenishment /*must be consistent everywhere for complete mdp defininition and associated policies.*/
	{
		class canOrderPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;

			// Import can order control parameters as vectors
			std::vector<int64_t> reorderPoint, canOrderPoint, orderUpToLevel;
			int64_t nrProducts, leadTime;

		public:
			canOrderPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class periodicReviewPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;

			// Import can order control parameters as vectors
			std::vector<int64_t> reviewPeriod, reorderPoint, orderUpToLevel;
			int64_t nrProducts, leadTime;

		public:
			periodicReviewPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

	}
}

