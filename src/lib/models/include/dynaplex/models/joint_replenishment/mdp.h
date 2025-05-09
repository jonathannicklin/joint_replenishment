#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace joint_replenishment {/*must be consistent everywhere for complete mdp definition and associated policies and states (if not defined inline).*/

		class MDP {

		public:
			// define variables separated in types
			double discountFactor, penaltyCost, holdingCost, orderCost, smoothingParameter, capacity;
			int64_t nrProducts, leadTime, maxPallets;
			bool isNonStationary;
			std::vector<double> initialForecast, initialSigma, volume, orderRate;
			std::vector<std::vector<double>> demandProb;

			// define an SKU class
			struct SKU {
				int64_t skuNumber;
				double forecastedDemand;
				double forecastDeviation;
				double inventoryLevel;
				std::vector<int64_t> orderQty;

				// assign SKU features to a DynaPlex::Features object
				void AddSKUToFeatures(DynaPlex::Features& feats) const {
					feats.Add(forecastedDemand);
					feats.Add(forecastDeviation);
					feats.Add(inventoryLevel);
					feats.Add(orderQty);
				}

				// converts SKU data into a DynaPlex::VarGroup object
				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vars;
					vars.Add("skuNumber", skuNumber);
					vars.Add("forecastedDemand", forecastedDemand);
					vars.Add("forecastDeviation", forecastDeviation);
					vars.Add("inventoryLevel", inventoryLevel);
					vars.Add("orderQty", orderQty);
					return vars;
				}

				// initialise an empty SKU object
				SKU() {}

				// initialises an SKU object using data from DynaPlex::VarGroup
				explicit SKU(const DynaPlex::VarGroup& vars) {
					vars.Get("skuNumber", skuNumber);
					vars.Get("forecastedDemand", forecastedDemand);
					vars.Get("forecastDeviation", forecastDeviation);
					vars.Get("inventoryLevel", inventoryLevel);
					vars.Get("orderQty", orderQty);
				}

				// compares two SKU objects for equality
				bool operator==(const SKU& other) const = default;
			};

			// defines actionsList variable as a integer vector
			std::vector<int64_t> actionsList;

			struct State {
				DynaPlex::StateCategory cat;

				// defines SKUs as a vector
				std::vector<SKU> SKUs;

				double usedCapacity;
				int64_t remainingEvents;
				float periodOrderingCosts;
				float periodBackorderCosts;
				float periodHoldingCosts;
				int64_t orderItem;
				int64_t periodCount = 1;
				// float discountFactor;

				// convert State object into a DynaPlex::VarGroup object
				DynaPlex::VarGroup ToVarGroup() const;

				// checks if two state objects are equal -- note; does not always work and can be removed 
				bool operator==(const State& other) const = default;
			};

			// enumeration class which defines a set of integer constants
			enum class actionType :int64_t {
				productSelection,
				quantitySelection,
				pass
			};

			// creates a structure to to represent specific actions with a number
			struct actionClass {
				actionType type;
				int64_t number;
			};

			// defines a vector to contain actions
			std::vector<actionClass> actions;

			// defines the variable Event to be a vector of integers
			using Event = std::vector<double>; // incoming demand in each new period expressed in pallets

			// remainder of DynaPlex API
			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;
			Event GetEvent(DynaPlex::RNG& rng) const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State& state, int64_t action) const;
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			// enables MDPs to be uniformly constructed
			explicit MDP(const DynaPlex::VarGroup&);
		};
	}
}

