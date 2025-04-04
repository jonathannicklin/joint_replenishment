﻿#include <iostream>
#include <fstream>
#include <vector>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/retrievestate.h"
#include "dynaplex/models/joint_replenishment/mdp.h"

using namespace DynaPlex;

void exportToCSV(const std::vector<std::vector<int64_t>>& data, const std::string& filename) {
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file is open
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // Loop through each row and each column to write data into CSV format
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            // Write the value followed by a comma, but no comma after the last element in a row
            outFile << data[i][j];

            // Check if this is not the last column, then add a comma
            if (j != data[i].size() - 1) {
                outFile << ";";
            }
        }
        // After each row, add a new line
        outFile << "\n";
    }

    // Close the file after writing
    outFile.close();
}

int train() {

    for (int64_t k = 0; k < 1; k++) { // mdp configuration loop

        // init
        auto& dp = DynaPlexProvider::Get();
        std::string model_name = "joint_replenishment"; // define model name
        std::string mdp_config_name = "mdp_config_" + std::to_string(k) + ".json"; // add model configurations
        std::string mdp_config_name_testdata = "mdp_config_0.json"; // set up test demand data (change if more mdp environments are tested)

        // define mdp training environment
        std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name);
        auto mdp_vars_from_json = DynaPlex::VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);

        // define policy used to train neural network
        auto initialPolicy = mdp->GetPolicy("random");

        // set DCL parameters
        DynaPlex::VarGroup nn_training{
                {"early_stopping_patience",10},
                // { "max_training_epochs", 200 }
        };

        DynaPlex::VarGroup nn_architecture{
                {"type","mlp"},
                {"hidden_layers",DynaPlex::VarGroup::Int64Vec{128,128}}
        };

        int64_t num_gens = 4;
        DynaPlex::VarGroup dcl_config{
            {"N",20000}, // number of samples
            {"num_gens",num_gens},
            {"M",2000}, // number of exogenous samples
            {"H", 500},// rollout horizon length
            {"L", 0},// warm-up period
            {"nn_architecture",nn_architecture},
            {"nn_training",nn_training},
            {"retrain_lastgen_only",false}
            // {"reinitiate_counter", 200} // ensures not stuck in state
        };

        try {
            // train policy
            auto dcl = dp.GetDCL(mdp, initialPolicy, dcl_config);
            dcl.TrainPolicy(); // saves to disk

            // get all trained policies, as well as the initial policy, in vector form (generations)
            auto policies = dcl.GetPolicies();

            for (int64_t k = 0; k < 2; k++) {

                // retrieve policy to evaluate
                std::string policy_config_name = "policy_config_" + std::to_string(k) + ".json"; // 1 = can order 2 = periodic review
                std::string policy_file_path = dp.System().filepath("mdp_config_examples", model_name, policy_config_name);
                VarGroup evaluatePolicy = VarGroup::LoadFromFile(policy_file_path);
                auto policy = mdp->GetPolicy(evaluatePolicy);
                policies.push_back(policy);
            }

            // define test environment
            std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name_testdata);
            mdp_vars_from_json = DynaPlex::VarGroup::LoadFromFile(file_path);
            auto mdp_test = dp.GetMDP(mdp_vars_from_json);

            // define comparison parameters
            VarGroup comparerConfig;
            comparerConfig.Add("number_of_trajectories", 1000);
            comparerConfig.Add("periods_per_trajectory", 1040);

            // compare policies
            auto comparer = dp.GetPolicyComparer(mdp_test, comparerConfig);
            auto comparison = comparer.Compare(policies);
            for (auto& VarGroup : comparison) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }

        catch (const std::exception& e) {
            std::cout << "exception: " << e.what() << std::endl;
        }
    }

    return 0;
}

int evaluate() {

    try {
        auto& dp = DynaPlexProvider::Get();
        DynaPlex::VarGroup config; // may not need
        auto& system = dp.System();
        std::string model_name = "joint_replenishment";
        std::string mdp_config_name = "mdp_config_0.json"; // define the mdp configuration you want to evaluate here
        std::string file_path = system.filepath("mdp_config_examples", model_name, mdp_config_name);

        // define mdp environment
        auto mdp_vars_from_json = VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);
        using underlying_mdp = DynaPlex::Models::joint_replenishment::MDP;
        std::int64_t rng_seed = 11112014;
        Trajectory trajectory{};
        trajectory.RNGProvider.SeedEventStreams(true, rng_seed);
        int64_t sampleNumber = 10400; // define sample horizon

        bool ruleBasedEvaluate = false;
        bool dclEvaluate = true;
        bool ppoEvaluate = false;

        if (ruleBasedEvaluate == true) {

            // initiate state and state policies to evaluate
            // std::vector<std::string> evaluatePolicyList = { "canOrderPolicy", "periodicReviewPolicy" };
            std::vector<std::string> evaluatePolicyList = { "canOrderPolicy"};

            for (int64_t k = 0; k < evaluatePolicyList.size(); k++) {

                // initiate state
                mdp->InitiateState({ &trajectory,1 });

                // retrieve policy to evaluate
                std::string policy_config_name = "policy_config_" + std::to_string(k) + ".json"; // 1 = can order 2 = periodic review
                std::string policy_file_path = dp.System().filepath("mdp_config_examples", model_name, policy_config_name);
                VarGroup evaluatePolicy = VarGroup::LoadFromFile(policy_file_path);
                auto policy = mdp->GetPolicy(evaluatePolicy);

                // import data from mdp setup
                int64_t containerVolume;
                int64_t penaltyCost;
                int64_t nrProducts;
                int64_t leadTime;

                mdp_vars_from_json.Get("capacity", containerVolume);
                mdp_vars_from_json.Get("penaltyCost", penaltyCost);
                mdp_vars_from_json.Get("nrProducts", nrProducts);
                mdp_vars_from_json.Get("leadTime", leadTime);

                // containers for graphing
                std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
                std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

                // init tracking KPIs
                float serviceLevel = 0;
                float fillRate = 0;
                float periodicity = 0;
                int64_t backorderCost = 0;
                int64_t holdingCost = 0;
                int64_t orderingCost = 0;
                int64_t unmetDemand = 0;
                int64_t demand = 0;
                int64_t totalDemand = 0;
                int64_t inventoryPosition = 0;
                std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
                std::vector<int64_t> backorderPrior(nrProducts, 0);
                std::vector<int64_t> backorderPost(nrProducts, 0);
                std::vector<int64_t> orderQtyPrior(nrProducts, 0);
                bool final_reached = false;

                // track and calculate KPIs
                for (int64_t i = 0; i < sampleNumber;) {

                    auto& cat = trajectory.Category;
                    auto& state = DynaPlex::RetrieveState<underlying_mdp::State>(trajectory.GetState());

                    if (cat.IsAwaitEvent()) {
                        fillRate += state.usedCapacity / containerVolume; // later divide by number of orders

                        if (state.usedCapacity != 0) {
                            periodicity += 1; // here we define # orders; later divide by number of sample periods to get periodicity
                        }

                        for (int64_t j = 0; j < nrProducts; j++) { // store inventory position level before demand
                            auto& product = state.SKUs[j];

                            inventoryLevelPrior[j] = product.inventoryLevel;
                            orderQtyPrior[j] = product.orderQty[0]; // when event is incorporated, we lose this information
                        }

                        mdp->IncorporateEvent({ &trajectory,1 });

                        orderingCost += state.periodOrderingCosts;
                        holdingCost += state.periodHoldingCosts;
                        backorderCost += state.periodBackorderCosts;

                        for (int64_t j = 0; j < nrProducts; j++) { // derive demand from previous and current inventory levels
                            auto& product = state.SKUs[j];

                            demand = inventoryLevelPrior[j] - (product.inventoryLevel - orderQtyPrior[j]);
                            totalDemand += demand;

                            if (demand != 0) {
                                if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] > 0) {
                                    unmetDemand += demand - inventoryLevelPrior[j];
                                }
                                else if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] <= 0) {
                                    unmetDemand += demand;
                                }
                            }

                            inventoryPosition = product.inventoryLevel;
                            for (int64_t k = 0; k < leadTime; k++) {
                                inventoryPosition += product.orderQty[k];
                            }

                            inventoryPositionsOverTime[j][i] = inventoryPosition;
                            inventoryLevelsOverTime[j][i] = product.inventoryLevel;
                        }
                        i++; // count number of sample periods
                    }

                    else if (cat.IsAwaitAction()) {
                        policy->SetAction({ &trajectory,1 });
                        mdp->IncorporateAction({ &trajectory,1 });
                    }

                    else if (cat.IsFinal()) {
                        final_reached = true;
                    }
                }

                // determine final values
                holdingCost = holdingCost / sampleNumber;
                backorderCost = backorderCost / sampleNumber;
                orderingCost = orderingCost / sampleNumber;
                fillRate = fillRate / periodicity; // periodicity currently used as order counter
                periodicity = periodicity / sampleNumber;
                serviceLevel = (static_cast<float>(totalDemand) - unmetDemand) / totalDemand; // change

                std::cout << "Total demand: " << totalDemand << std::endl;
                std::cout << "Total demand met: " << totalDemand - unmetDemand << std::endl;

                // store variables after evaluation
                VarGroup kpis{};
                kpis.Add("serviceLevel", serviceLevel);
                kpis.Add("fillRate", fillRate);
                kpis.Add("periodicity", periodicity);
                kpis.Add("backorderCost", backorderCost);
                kpis.Add("holdingCost", holdingCost);
                kpis.Add("orderingCost", orderingCost);
                kpis.Add("totalCost", trajectory.CumulativeReturn / sampleNumber);

                // output evaluation to a file
                auto filename = dp.System().filepath("joint_replenishment", "Evaluation", evaluatePolicyList[k] + "_evaluation.json");
                kpis.SaveToFile(filename);

                // export containers for graphing
                exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_" + evaluatePolicyList[k] + ".csv");
                exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_" + evaluatePolicyList[k] + ".csv");
            }

        }

        if (dclEvaluate == true) {

            // get trained DCL policy
            std::string gen = "10"; // change to correct gen
            std::cout << mdp->Identifier() << std::endl;
            auto filename = dp.System().filepath(mdp->Identifier(), "dcl_policy_gen" + gen); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            auto dclPolicy = dp.LoadPolicy(mdp, path);

            // initiate state
            mdp->InitiateState({ &trajectory,1 });

            // import data from mdp setup
            int64_t containerVolume;
            int64_t penaltyCost;
            int64_t nrProducts;
            int64_t leadTime;

            mdp_vars_from_json.Get("capacity", containerVolume);
            mdp_vars_from_json.Get("penaltyCost", penaltyCost);
            mdp_vars_from_json.Get("nrProducts", nrProducts);
            mdp_vars_from_json.Get("leadTime", leadTime);

            // containers for graphing
            std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
            std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

            // init tracking KPIs
            float serviceLevel = 0;
            float fillRate = 0;
            float periodicity = 0;
            int64_t backorderCost = 0;
            int64_t holdingCost = 0;
            int64_t orderingCost = 0;
            int64_t unmetDemand = 0;
            int64_t demand = 0;
            int64_t totalDemand = 0;
            int64_t inventoryPosition = 0;
            std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
            std::vector<int64_t> backorderPrior(nrProducts, 0);
            std::vector<int64_t> backorderPost(nrProducts, 0);
            std::vector<int64_t> orderQtyPrior(nrProducts, 0);
            bool final_reached = false;

            // track and calculate KPIs
            for (int64_t i = 0; i < sampleNumber;) {

                auto& cat = trajectory.Category;
                auto& state = DynaPlex::RetrieveState<underlying_mdp::State>(trajectory.GetState());

                if (cat.IsAwaitEvent()) {
                    fillRate += state.usedCapacity / containerVolume; // later divide by number of orders

                    if (state.usedCapacity != 0) {
                        periodicity += 1; // here we define # orders; later divide by number of sample periods to get periodicity
                    }

                    for (int64_t j = 0; j < nrProducts; j++) { // store inventory position level before demand
                        auto& product = state.SKUs[j];

                        inventoryLevelPrior[j] = product.inventoryLevel;
                        orderQtyPrior[j] = product.orderQty[0]; // when event is incorporated, we lose this information
                    }

                    mdp->IncorporateEvent({ &trajectory,1 });

                    orderingCost += state.periodOrderingCosts;
                    holdingCost += state.periodHoldingCosts;
                    backorderCost += state.periodBackorderCosts;

                    for (int64_t j = 0; j < nrProducts; j++) { // derive demand from previous and current inventory levels
                        auto& product = state.SKUs[j];

                        demand = inventoryLevelPrior[j] - (product.inventoryLevel - orderQtyPrior[j]);
                        totalDemand += demand;

                        if (demand != 0) {
                            if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] > 0) {
                                unmetDemand += demand - inventoryLevelPrior[j];
                            }
                            else if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] <= 0) {
                                unmetDemand += demand;
                            }
                        }

                        inventoryPosition = product.inventoryLevel;
                        for (int64_t k = 0; k < leadTime; k++) {
                            inventoryPosition += product.orderQty[k];
                        }

                        inventoryPositionsOverTime[j][i] = inventoryPosition;
                        inventoryLevelsOverTime[j][i] = product.inventoryLevel;
                    }
                    i++; // count number of sample periods
                }

                else if (cat.IsAwaitAction()) {
                    dclPolicy->SetAction({ &trajectory,1 });
                    mdp->IncorporateAction({ &trajectory,1 });
                }

                else if (cat.IsFinal()) {
                    final_reached = true;
                }
            }

            // determine final values
            holdingCost = holdingCost / sampleNumber;
            backorderCost = backorderCost / sampleNumber;
            orderingCost = orderingCost / sampleNumber;
            fillRate = fillRate / periodicity; // periodicity currently used as order counter
            periodicity = periodicity / sampleNumber;
            serviceLevel = (static_cast<float>(totalDemand) - unmetDemand) / totalDemand; // change

            std::cout << "Total demand: " << totalDemand << std::endl;
            std::cout << "Total demand met: " << totalDemand - unmetDemand << std::endl;

            // store variables after evaluation
            VarGroup kpis{};
            kpis.Add("serviceLevel", serviceLevel);
            kpis.Add("fillRate", fillRate);
            kpis.Add("periodicity", periodicity);
            kpis.Add("backorderCost", backorderCost);
            kpis.Add("holdingCost", holdingCost);
            kpis.Add("orderingCost", orderingCost);
            kpis.Add("totalCost", trajectory.CumulativeReturn / sampleNumber);

            // output evaluation to a file
            filename = dp.System().filepath("joint_replenishment", "Evaluation", "dcl_evaluation.json");
            kpis.SaveToFile(filename);

            // export containers for graphing
            exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_dcl.csv");
            exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_dcl.csv");
        }

        if (ppoEvaluate == true) {

            // get trained ppo policy
            // std::string gen = "2"; // change to correct gen
            auto filename = dp.System().filepath(mdp->Identifier(), "ppo_policy"); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            auto ppoPolicy = dp.LoadPolicy(mdp, path);

            // initiate state
            mdp->InitiateState({ &trajectory,1 });

            // import data from mdp setup
            int64_t containerVolume;
            int64_t penaltyCost;
            int64_t nrProducts;
            int64_t leadTime;

            mdp_vars_from_json.Get("capacity", containerVolume);
            mdp_vars_from_json.Get("penaltyCost", penaltyCost);
            mdp_vars_from_json.Get("nrProducts", nrProducts);
            mdp_vars_from_json.Get("leadTime", leadTime);

            // containers for graphing
            std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
            std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

            // init tracking KPIs
            float serviceLevel = 0;
            float fillRate = 0;
            float periodicity = 0;
            int64_t backorderCost = 0;
            int64_t holdingCost = 0;
            int64_t orderingCost = 0;
            int64_t unmetDemand = 0;
            int64_t demand = 0;
            int64_t totalDemand = 0;
            int64_t inventoryPosition = 0;
            std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
            std::vector<int64_t> backorderPrior(nrProducts, 0);
            std::vector<int64_t> backorderPost(nrProducts, 0);
            std::vector<int64_t> orderQtyPrior(nrProducts, 0);
            bool final_reached = false;

            // track and calculate KPIs
            for (int64_t i = 0; i < sampleNumber;) {

                auto& cat = trajectory.Category;
                auto& state = DynaPlex::RetrieveState<underlying_mdp::State>(trajectory.GetState());

                if (cat.IsAwaitEvent()) {
                    fillRate += state.usedCapacity / containerVolume; // later divide by number of orders

                    if (state.usedCapacity != 0) {
                        periodicity += 1; // here we define # orders; later divide by number of sample periods to get periodicity
                    }

                    for (int64_t j = 0; j < nrProducts; j++) { // store inventory position level before demand
                        auto& product = state.SKUs[j];

                        inventoryLevelPrior[j] = product.inventoryLevel;
                        orderQtyPrior[j] = product.orderQty[0]; // when event is incorporated, we lose this information
                    }

                    mdp->IncorporateEvent({ &trajectory,1 });

                    orderingCost += state.periodOrderingCosts;
                    holdingCost += state.periodHoldingCosts;
                    backorderCost += state.periodBackorderCosts;

                    for (int64_t j = 0; j < nrProducts; j++) { // derive demand from previous and current inventory levels
                        auto& product = state.SKUs[j];

                        demand = inventoryLevelPrior[j] - (product.inventoryLevel - orderQtyPrior[j]);
                        totalDemand += demand;

                        if (demand != 0) {
                            if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] > 0) {
                                unmetDemand += demand - inventoryLevelPrior[j];
                            }
                            else if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] <= 0) {
                                unmetDemand += demand;
                            }
                        }

                        inventoryPosition = product.inventoryLevel;
                        for (int64_t k = 0; k < leadTime; k++) {
                            inventoryPosition += product.orderQty[k];
                        }

                        inventoryPositionsOverTime[j][i] = inventoryPosition;
                        inventoryLevelsOverTime[j][i] = product.inventoryLevel;
                    }
                    i++; // count number of sample periods
                }

                else if (cat.IsAwaitAction()) {
                    ppoPolicy->SetAction({ &trajectory,1 });
                    mdp->IncorporateAction({ &trajectory,1 });
                }

                else if (cat.IsFinal()) {
                    final_reached = true;
                }
            }

            // determine final values
            holdingCost = holdingCost / sampleNumber;
            backorderCost = backorderCost / sampleNumber;
            orderingCost = orderingCost / sampleNumber;
            fillRate = fillRate / periodicity; // periodicity currently used as order counter
            periodicity = periodicity / sampleNumber;
            serviceLevel = (static_cast<float>(totalDemand) - unmetDemand) / totalDemand; // change

            std::cout << "Total demand: " << totalDemand << std::endl;
            std::cout << "Total demand met: " << totalDemand - unmetDemand << std::endl;

            // store variables after evaluation
            VarGroup kpis{};
            kpis.Add("serviceLevel", serviceLevel);
            kpis.Add("fillRate", fillRate);
            kpis.Add("periodicity", periodicity);
            kpis.Add("backorderCost", backorderCost);
            kpis.Add("holdingCost", holdingCost);
            kpis.Add("orderingCost", orderingCost);
            kpis.Add("totalCost", trajectory.CumulativeReturn / sampleNumber);

            // output evaluation to a file
            filename = dp.System().filepath("joint_replenishment", "Evaluation", "ppo_evaluation.json");
            kpis.SaveToFile(filename);

            // export containers for graphing
            exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_ppo.csv");
            exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_ppo.csv");
        }
    }

    catch (const DynaPlex::Error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

int heatMap() {
    try {

        // get model information
        auto& dp = DynaPlexProvider::Get();
        // auto& system = dp.System();
        std::string model_name = "joint_replenishment";
        std::string mdp_config_name = "mdp_config_0.json";

        std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name);
        auto mdp_vars_from_json = VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);

        auto policy = mdp->GetPolicy("random");
        
        bool ruleBasedEvaluate = false;
        bool dclEvaluate = true;
        bool ppoEvaluate = false;

        if (ruleBasedEvaluate == true) {

        }

        if (dclEvaluate == true) {
            // get trained DCL policy
            std::string gen = "10"; // change to correct gen
            std::cout << mdp->Identifier() << std::endl;
            auto filename = dp.System().filepath(mdp->Identifier(), "dcl_policy_gen" + gen); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            policy = dp.LoadPolicy(mdp, path); // udpate policy from random
        }

        if (ppoEvaluate == true) {

        }

        //get some initializing variables from json
        std::vector<int64_t> initialForecast;
        std::vector<int64_t> initialSigma;
        int64_t nrProducts;
        int64_t leadTime;

        mdp_vars_from_json.Get("initialForecast", initialForecast);
        mdp_vars_from_json.Get("initialSigma", initialSigma);
        mdp_vars_from_json.Get("nrProducts", nrProducts);
        mdp_vars_from_json.Get("leadTime", leadTime);

        // define heat map parameters
        int64_t item1 = 0;
        int64_t item2 = 1;
        int64_t maxInventoryPosition = 60; // set by looking at inventory plot

        // define container variables
        std::vector<int64_t> actions;

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


        for (int inventoryPosition1 = 0; inventoryPosition1 < maxInventoryPosition; inventoryPosition1++) {
            for (int inventoryPosition2 = 0; inventoryPosition2 < maxInventoryPosition; inventoryPosition2++) {
                auto SKUs = std::vector<SKU>{};
                SKUs.reserve(nrProducts);

                // creates multiple SKU objects equal to nrProducts
                for (size_t i = 0; i < nrProducts; i++) {
                    SKU item{};
                    item.skuNumber = i;
                    item.forecastedDemand = initialForecast[i];
                    item.forecastDeviation = initialSigma[i];
                    if (i == item1) {
                        item.inventoryLevel = inventoryPosition1;
                    }
                    else if (i == item2) {
                        item.inventoryLevel = inventoryPosition2;
                    }
                    else {
                        item.inventoryLevel = initialForecast[i];
                    }
                    item.orderQty.resize(leadTime, 0);
                    SKUs.push_back(item);
                }
                
                DynaPlex::VarGroup stateVars{
                    {"cat", StateCategory::AwaitAction().ToVarGroup()}, // change this to ordering 
                    {"usedCapacity", 0},
                    {"SKUs", SKUs},
                    {"remainingEvents", 100},
                    {"orderItem", item2},
                    {"periodOrderingCosts", 0},
                    {"periodHoldingCosts", 0},
                    {"periodBackorderCosts", 0},
                    {"periodCount", 1000}
                };

                auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                //state

                std::vector<Trajectory> trajVec{};

                mdp->InitiateState({ &trajVec[0] ,1 }, state);
                policy->SetAction(trajVec);

                actions.push_back(trajVec[0].NextAction);
            };

            //just an example of storing the variables after evaluation
            VarGroup kpis{};
            kpis.Add("actions", actions);

            auto filename = dp.System().filepath("joint_replenishment", "Evaluation", "heatmap_items" + std::to_string(item1) + std::to_string(item2) + ".json");
            kpis.SaveToFile(filename);
        }
    }

    catch (const DynaPlex::Error& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

int main() {

    // train();
    // evaluate();
    heatMap();
};
