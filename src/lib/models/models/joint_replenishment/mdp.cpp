// #include "mdp.h"
#include "dynaplex/models/joint_replenishment/mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace joint_replenishment { /*keep this in line with id below and with namespace name in header*/

		// collects and returns static information about the MDP in a VarGroup object
		VarGroup MDP::GetStaticInfo() const {
			VarGroup vars;

			// determine valid actions, demand type and horizon considered
			int64_t validActions = actions.size(); // possible actions are given within actions vector, each corresponding to a number
			vars.Add("valid_actions", validActions);
			vars.Add("isNonStationary", isNonStationary);
			if (isNonStationary) {
				vars.Add("horizon_type", "finite");
			}
			else {
				vars.Add("horizon_type", "infinite");
			}

			return vars;
		}

		// calculates period rewards, updates forecasts, initialises for new period
		double MDP::ModifyStateWithEvent(State& state, const Event& event) const {
			double reward = 0.0;
			state.periodHoldingCosts = 0;
			state.periodBackorderCosts = 0;
			state.periodOrderingCosts = 0;
			bool order = false;

			for (int productID = 0; productID < nrProducts; productID++) {
				auto& product = state.SKUs[productID];
				auto demand = event[product.skuNumber];

				// determine if container was ordered
				if (product.orderQty[leadTime - 1] > 0) {
					order = true;
				}

				// update inventory levels
				product.inventoryLevel += product.orderQty[0] - demand;

				// update orderQty array
				for (size_t j = 0; j < leadTime - 1; ++j) {
					product.orderQty[j] = product.orderQty[j + 1];
				}
				product.orderQty[leadTime - 1] = 0; // set last element equal to zero

				// update forecast
				product.forecastedDemand = product.forecastedDemand / leadTime; // correct to single-period forecast
				product.forecastDeviation = product.forecastDeviation / sqrt(leadTime);

				product.forecastDeviation = sqrt(smoothingParameter * pow(demand - product.forecastedDemand, 2) +
					(1.0 - smoothingParameter) * product.forecastDeviation * product.forecastDeviation);

				product.forecastedDemand = smoothingParameter * demand +
					(1.0 - smoothingParameter) * product.forecastedDemand;

				product.forecastedDemand = product.forecastedDemand * leadTime; // convert back to forecast over lead time duration
				product.forecastDeviation = product.forecastDeviation * sqrt(leadTime);

				// calculate reward
				if (product.inventoryLevel >= 0) {
					reward += holdingCost * std::ceil(product.inventoryLevel); // pallet costs are per pallet space, thus rounded up
					state.periodHoldingCosts += holdingCost * std::ceil(product.inventoryLevel);
				}
				else {
					reward += penaltyCost * (-product.inventoryLevel);
					state.periodBackorderCosts += penaltyCost * (-product.inventoryLevel);
					// product.inventoryLevel = 0; // lost sales instead of backordering
				}
			}

			// calculate ordering costs
			if (order == true) {
				state.periodOrderingCosts += orderCost;
			}

			// reset for new period
			state.usedCapacity = 0;
			// state.periodOrderingCosts = 0;

			// move in transit inventory one step ahead in time
			state.cat = StateCategory::AwaitAction(0);

			if (isNonStationary) {
				state.remainingEvents -= 1;
			}

			if (state.remainingEvents == 0) {
				state.cat = StateCategory::Final();
			}

			state.periodCount += 1;

			return reward;
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const {
			if (!IsAllowedAction(state, action)) {
				DynaPlex::VarGroup stateInfo = state.ToVarGroup();
				std::cout << stateInfo.Dump() << std::endl;
				std::cout << "this action (" << action << ") should not be allowed.";
				throw DynaPlex::Error("joint_replenishment :: action not allowed; state.usedCapacity");
			}
			auto& [type, index] = actions[action]; // action types: (1) product selection and (2) quantitiy selection; index refers to decision of action type!
			double remaining = capacity - state.usedCapacity;

			switch (type) {
			case actionType::pass:
				state.cat = StateCategory::AwaitEvent();
				return 0.0;

			case actionType::productSelection: { // state is not updated after actionType::productSelection

				// pass if there is not enough capacity to facilitate ordering a single pallet
				if (volume[index] > remaining) {
					state.cat = StateCategory::AwaitEvent();
				}

				// move onto quantitySelection
				else {
					// update product being ordered, necessary to connect decision stages
					state.orderItem = index;
					state.cat = StateCategory::AwaitAction(1); // quantitySelection
				}
				return 0.0;
			}

			case actionType::quantitySelection: {
				state.SKUs[state.orderItem].orderQty[leadTime - 1] = index; // update state to reflect chosen orderQty
				state.cat = StateCategory::AwaitAction(0); // productSelection
				state.usedCapacity += index * volume[state.orderItem]; // convert number of items ordered into volume
				return 0.0;
			}

			default:
				throw DynaPlex::Error("joint_replenishment :: invalid type in IsAllowedAction");
			}

			return 0.0; //returns costs [have not included within this, no sku-related order costs necessary?]
		}

		// set state information
		DynaPlex::VarGroup MDP::State::ToVarGroup() const {
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("usedCapacity", usedCapacity);
			vars.Add("SKUs", SKUs);
			vars.Add("remainingEvents", remainingEvents);
			vars.Add("orderItem", orderItem);
			return vars;
		}

		// get state information
		MDP::State MDP::GetState(const VarGroup& vars) const {
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("usedCapacity", state.usedCapacity);
			vars.Get("SKUs", state.SKUs);
			vars.Get("remainingEvents", state.remainingEvents);
			vars.Get("orderItem", state.orderItem);
			return state;
		}

		// initialise state variables at start of horizon
		MDP::State MDP::GetInitialState() const {
			State state{};
			state.SKUs = std::vector<SKU>{};
			state.SKUs.reserve(nrProducts);

			// creates multiple SKU objects equal to nrProducts
			for (size_t i = 0; i < nrProducts; i++) {
				SKU item{};
				item.skuNumber = i;
				item.forecastedDemand = initialForecast[i];
				item.forecastDeviation = initialSigma[i];
				item.inventoryLevel = std::ceil(initialForecast[i] + 1.282 * initialSigma[i]); // base stock 90%; rounded up
				item.orderQty.resize(leadTime, 0);
				state.SKUs.push_back(item);
			}

			// check that the number of elements in input arrays match nrProducts
			if (initialForecast.size() != nrProducts) {
				throw DynaPlex::Error("joint_replenishment :: initial forecast data does not match the number of items");
			}
			if (initialSigma.size() != nrProducts) {
				throw DynaPlex::Error("joint_replenishment :: initial sigma data does not match the number of items");
			}
			if (orderRate.size() != nrProducts) {
				throw DynaPlex::Error("joint_replenishment :: orderRate data does not match the number of items");
			}
			if (volume.size() != nrProducts) {
				throw DynaPlex::Error("joint_replenishment :: pallet volume data does not match the number of items");
			}

			// start by selecting a product to produce
			state.cat = StateCategory::AwaitAction(0);

			state.orderItem = 0;
			state.usedCapacity = 0;
			state.remainingEvents = 100;

			return state;
		}

		// sets all static information when initialising MDP from mdp.h
		MDP::MDP(const VarGroup& config) {

			// init state variables
			config.Get("discountFactor", discountFactor);
			config.Get("penaltyCost", penaltyCost);
			config.Get("holdingCost", holdingCost);
			config.Get("orderCost", orderCost);
			config.Get("orderRate", orderRate);
			config.Get("leadTime", leadTime);
			config.Get("initialForecast", initialForecast);
			config.Get("initialSigma", initialSigma);
			config.Get("volume", volume);
			config.Get("smoothingParameter", smoothingParameter);
			config.Get("capacity", capacity);
			config.Get("nrProducts", nrProducts);
			config.Get("isNonStationary", isNonStationary);
			config.Get("maxPallets", maxPallets);

			// build demandProb 2D array
			for (int64_t i = 0; i < nrProducts; i++) {
				std::vector<double> buildDemandArray(10, 0); // assuming that no more than 10 pallets ordered at once
				config.Get("SKU_" + std::to_string(i), buildDemandArray);
				demandProb.push_back(buildDemandArray);
			}

			// create actions list
			for (int64_t i = 0; i < nrProducts; i++) {
				actions.push_back({ actionType::productSelection,i });
				actionsList.push_back(i);
			}

			for (int64_t i = 1; i <= maxPallets; i++) {
				actionsList.push_back(i);
				actions.push_back({ actionType::quantitySelection, i });
			}

			actions.push_back({ actionType::pass,0 });
			actionsList.push_back(0);
		}

		// generate demand vector
		MDP::Event MDP::GetEvent(RNG& rng) const {
			// generate demand for each product
			std::vector<double> demandVector; // demand expressed in partial pallets
			demandVector.reserve(nrProducts);

			for (int64_t product = 0; product < nrProducts; ++product) { // simulates compound poisson demand for each product
				double demand;

				// simulate compound Poisson demand
				if (DiscreteDist::GetCustomDist({ 1 - orderRate[product], orderRate[product] }).GetSample(rng) == 1) {
					if (isNonStationary) {
						throw DynaPlex::Error("joint_replenishment :: non stationary demand not implemented");
					}
					else {
						demand = DiscreteDist::GetCustomDist(demandProb[product], 1).GetSample(rng);
					}
				}
				else {
					demand = 0;
				}
				demandVector.push_back(demand);
			}

			return demandVector;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {
			for (size_t i = 0; i < state.SKUs.size();i++) {
				state.SKUs[i].AddSKUToFeatures(features);
			}

			features.Add(state.usedCapacity);
			if (isNonStationary) {
				features.Add(state.remainingEvents);
			}
		}

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const {

			registry.Register<canOrderPolicy>("canOrderPolicy",
				"This policy is replicates following a rule-based can order policy. ");

			registry.Register<periodicReviewPolicy>("periodicReviewPolicy",
				"This policy is replicates following a rule-based periodic review order policy. ");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const {
			return state.cat;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {

			// give error when action chosen is not available within actions vector
			if (action < 0 || action >= actions.size()) {
				throw DynaPlex::Error("joint_replenishment :: invalid action index");
			}

			// retreive type, index, currentDecision, remaining data; init enoughSpace
			auto& [type, index] = actions[action];
			auto currentDecision = state.cat.Index(); // 0 = productSelection; 1 = quantitySelection
			double remaining = capacity - state.usedCapacity;

			switch (type) {
			case actionType::pass:
				return currentDecision == 0; // returns true if current decision is productSelection

			case actionType::productSelection:
				// check amount fits in remaining space
				if (volume[index] > remaining) {
					return false;
				}

				return currentDecision == 0 && state.SKUs[index].orderQty[leadTime - 1] == 0; // returns true if  current decision is productSelection, item not previously ordered

			case actionType::quantitySelection: {
				if (volume[state.orderItem] > remaining) {
					return false;
				}

				return currentDecision == 1 && state.SKUs[state.orderItem].orderQty[leadTime - 1] == 0; // returns true if current decision is quantitySelection, item not previously ordered
			}
			default:
				throw DynaPlex::Error("joint_replenishment :: invalid type in IsAllowedAction");
			}
		}

		void Register(DynaPlex::Registry& registry) {
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel("joint_replenishment", "Joint replenishment of items for an MSc Thesis at Ahold Delhaize", registry);
		}
	}
}