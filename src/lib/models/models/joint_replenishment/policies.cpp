#include "policies.h"
#include "dynaplex/models/joint_replenishment/mdp.h"
#include <cstdlib>
#include "dynaplex/error.h"

namespace DynaPlex::Models {
	namespace joint_replenishment  { /*keep this namespace name in line with the name space in which the mdp corresponding to this policy is defined*/
	
		canOrderPolicy::canOrderPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp } {
			config.Get("reorderPoint", reorderPoint);
			config.Get("canOrderPoint", canOrderPoint);
			config.Get("orderUpToLevel", orderUpToLevel);
		}

		int64_t canOrderPolicy::GetAction(const MDP::State& state) const {
			MDP::actionClass actionSelected{};
			int64_t actionIndex; // refers to the placement of action on actionList
			std::vector<int64_t> productsAllowed, inventoryPosition(mdp->nrProducts);

			if (state.cat.Index() == 0) { // product selection

				bool order = false;
				for (int64_t product = 0; product < mdp->nrProducts; ++product) {

					// calculate inventory position
					inventoryPosition[product] = state.SKUs[product].inventoryLevel;
					for (int64_t i = 0; i < mdp->leadTime;i++) {
						inventoryPosition[product] += state.SKUs[product].orderQty[i];
					}

					// check if an item tirggers
					if (inventoryPosition[product] <= reorderPoint[product]) {
						order = true;
					}
				}

				if (order == true) {

					// check which products have not already been ordered that need to be
					for (int64_t product = 0; product < mdp->nrProducts; product++) {
						if (state.SKUs[product].orderQty[mdp->leadTime - 1] == 0 && inventoryPosition[product] <= canOrderPoint[product]) {
							productsAllowed.push_back(product);
						}
					}

					// if all products have been ordered already, move onto new period
					if (productsAllowed.size() == 0) {
						actionSelected.type = MDP::actionType::pass; // always define (1) action type and (2) decision of action
						actionSelected.number = 0;
						actionIndex = mdp->nrProducts + mdp->maxPallets;
					}

					else {
						actionSelected.type = MDP::actionType::productSelection;
						int randomIndex = std::rand() % productsAllowed.size();
						actionSelected.number = productsAllowed[randomIndex];
						actionIndex = actionSelected.number;
					}
				}

				// no order placed
				else { 
					actionSelected.type = MDP::actionType::pass;
					actionSelected.number = 0;
					actionIndex = mdp->nrProducts + mdp->maxPallets;
				}
			}
			
			else if (state.cat.Index() == 1) { // quantity selection

				// calculate inventory position
				for (int64_t product = 0; product < mdp->nrProducts; product++) {

					inventoryPosition[product] = state.SKUs[product].inventoryLevel;
					for (int64_t i = 0; i < mdp->leadTime;i++) {
						inventoryPosition[product] += state.SKUs[product].orderQty[i];
					}
				}

				actionSelected.type = MDP::actionType::quantitySelection;
				actionSelected.number = orderUpToLevel[state.orderItem] - inventoryPosition[state.orderItem];
				
				if (actionSelected.number * mdp->volume[state.orderItem] + state.usedCapacity > mdp->capacity) {
					actionSelected.type = MDP::actionType::pass; // always define (1) action type and (2) decision of action
					actionSelected.number = 0;
					actionIndex = mdp->nrProducts + mdp->maxPallets;
				}
				else {
					actionIndex = mdp->nrProducts + actionSelected.number;
				}
			}

			else {
				throw DynaPlex::Error("joint_replenishment :: invalid aciton input in canOrderPolicy");
			}

			return actionIndex;
		}

		periodicReviewPolicy::periodicReviewPolicy(std::shared_ptr<const MDP> mdp, const DynaPlex::VarGroup& config)
			:mdp{ mdp } {
			config.Get("reviewPeriod", reviewPeriod);
			config.Get("reorderPoint", reorderPoint);
			config.Get("orderUpToLevel", orderUpToLevel);
		}

		int64_t periodicReviewPolicy::GetAction(const MDP::State& state) const {
			MDP::actionClass actionSelected{};
			int64_t actionIndex; // refers to placement of action on actionList
			std::vector<int64_t> productsAllowed, inventoryPosition(mdp->nrProducts);
			int64_t orderQuantity;

			if (state.cat.Index() == 0) { // product selection

				// check to see if we are in a review period
				if (state.periodCount % reviewPeriod[0] == 0) {

					// check which products have not already been ordered that need to be
					for (int64_t product = 0; product < mdp->nrProducts; product++) {

						// calculate inventory position
						inventoryPosition[product] = state.SKUs[product].inventoryLevel;
						for (int64_t i = 0; i < mdp->leadTime;i++) {
							inventoryPosition[product] += state.SKUs[product].orderQty[i];
						}

						// add to possible order items if (1) not already ordered (2) inventory position is below reorder point (3) can fit into remaining container space
						orderQuantity = orderUpToLevel[product] - inventoryPosition[product];
 						if (state.SKUs[product].orderQty[mdp->leadTime - 1] == 0 && inventoryPosition[product] <= reorderPoint[product] && state.usedCapacity + orderQuantity <= mdp-> capacity) {
							productsAllowed.push_back(product);
						}
					}

					// if all products have been ordered already, move onto new period
					if (productsAllowed.size() == 0) {
						actionSelected.type = MDP::actionType::pass; // always define (1) action type and (2) decision of action
						actionSelected.number = 0;
						actionIndex = mdp->nrProducts + mdp->maxPallets;
					}

					// randomly select from permissible items
					else {
						actionSelected.type = MDP::actionType::productSelection;
						int randomIndex = std::rand() % productsAllowed.size();
						actionSelected.number = productsAllowed[randomIndex];
						actionIndex = actionSelected.number;
					}
				}

				// not in review period for inventory; move to next period
				else {
					actionSelected.type = MDP::actionType::pass;
					actionSelected.number = 0;
					actionIndex = mdp->nrProducts + mdp->maxPallets;
				}
			}

			else if (state.cat.Index() == 1) { // quantity selection

				// calculate inventory position
				for (int64_t product = 0; product < mdp->nrProducts; product++) {

					inventoryPosition[product] = state.SKUs[product].inventoryLevel;
					for (int64_t i = 0; i < mdp->leadTime;i++) {
						inventoryPosition[product] += state.SKUs[product].orderQty[i];
					}
				}

				actionSelected.type = MDP::actionType::quantitySelection;
				actionSelected.number = orderUpToLevel[state.orderItem] - inventoryPosition[state.orderItem];

				if (actionSelected.number * mdp->volume[state.orderItem] + state.usedCapacity > mdp->capacity) {
					actionSelected.type = MDP::actionType::pass; // always define (1) action type and (2) decision of action
					actionSelected.number = 0;
					actionIndex = mdp->nrProducts + mdp->maxPallets;
				}
				else {
					actionIndex = mdp->nrProducts + actionSelected.number;
				}
			}

			else {
				throw DynaPlex::Error("joint_replenishment :: invalid action input in periodicReviewPolicy");
			}

			return actionIndex;
		}
	}
}