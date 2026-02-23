"""
Genetic Algorithm Optimizer for MA Crossover + RSI Strategy

This script uses genetic algorithm to optimize:
- Fast MA period
- Slow MA period
- Leverage multiplier

Strategy: MA Crossover with RSI(14) momentum filter
- Buy: Golden Cross AND RSI < 70 (avoid overbought)
- Sell: Death Cross AND RSI > 30 (avoid oversold)

Goal: Maximize Sharpe Ratio while controlling maximum drawdown
"""

import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_tick_data, preprocess_data, resample_to_kline
from src.strategy import MACrossoverStrategy
from src.backtest_engine import BacktestEngine
from src.metrics import PerformanceMetrics


# ============================================================================
# CONFIGURATION
# ============================================================================

# GA Parameters
POPULATION_SIZE = 64      # Population size
GENERATIONS = 128          # Number of generations
ELITE_SIZE = 16           # Number of elite individuals to preserve
MUTATION_RATE = 0.15      # Mutation probability
EARLY_STOP_PATIENCE = 20  # Stop if no improvement for N generations

# Gene Ranges
FAST_PERIOD_RANGE = (5, 30)      # Fast MA period range
SLOW_PERIOD_RANGE = (30, 100)    # Slow MA period range
LEVERAGE_RANGE = (1.0, 3.0)      # Leverage range

# Constraints
MAX_MDD = -15.0           # Maximum acceptable drawdown

# Data Configuration
TRAIN_DATA = "data/2024/XAUUSD_1y_24.csv"  # Training data
TEST_DATA = "data/2025/XAUUSD_1y_25.csv"   # Testing data
TIMEFRAME = '1h'          # K-line timeframe
INITIAL_CAPITAL = 10000


# ============================================================================
# Step 1: Define Individual Class
# ============================================================================

class Individual:
    """
    Represents an individual (a set of strategy parameters)

    Genes:
    - fast_period: Fast MA period
    - slow_period: Slow MA period
    - leverage: Leverage multiplier
    """

    def __init__(self, fast_period: int, slow_period: int, leverage: float):
        """Initialize individual"""
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.leverage = round(leverage, 2)

        # Performance metrics (populated after evaluation)
        self.fitness = 0.0
        self.ar = 0.0
        self.mdd = 0.0
        self.sharpe = 0.0
        self.sortino = 0.0

    def __repr__(self):
        """String representation of individual"""
        return (f"Individual(fast={self.fast_period}, slow={self.slow_period}, "
                f"lev={self.leverage}, fitness={self.fitness:.2f}, "
                f"AR={self.ar:.2f}%, MDD={self.mdd:.2f}%)")


# ============================================================================
# Step 2: Fitness Evaluation Function (CRITICAL)
# ============================================================================

def evaluate_fitness(individual: Individual, kline_data: pd.DataFrame) -> float:
    """
    Evaluate individual's fitness (score)

    Scoring criteria:
    1. If MDD exceeds limit -> score = 0 (eliminate)
    2. Otherwise -> score = Sharpe Ratio

    Args:
        individual: Individual to evaluate
        kline_data: K-line data (already resampled)

    Returns:
        fitness score (higher is better)
    """
    try:
        # Copy data to avoid modifying original
        kline = kline_data.copy()

        # Phase 3: Calculate indicators and signals
        strategy = MACrossoverStrategy(
            fast_period=individual.fast_period,
            slow_period=individual.slow_period
        )
        kline = strategy.calculate_indicators(kline, verbose=False)

        # Phase 4: Run backtest
        engine = BacktestEngine(leverage=individual.leverage)
        kline = engine.run(kline, verbose=False)

        # Phase 5: Calculate performance metrics
        metrics_calc = PerformanceMetrics(initial_capital=INITIAL_CAPITAL)
        metrics = metrics_calc.calculate(kline, verbose=False)

        # Store metrics in individual
        individual.ar = metrics['annual_return']
        individual.mdd = metrics['max_drawdown']
        individual.sharpe = metrics['sharpe_ratio']
        individual.sortino = metrics['sortino_ratio']

        # Calculate fitness score
        # Rule: Eliminate if MDD exceeds limit, otherwise use Sharpe Ratio as score
        if individual.mdd < MAX_MDD:
            individual.fitness = 0.0  # Risk too high, eliminate
        else:
            individual.fitness = individual.sharpe  # Use Sharpe Ratio as score

        return individual.fitness

    except Exception as e:
        # If error occurs (e.g., invalid parameters), set score to 0
        print(f"Error evaluating {individual}: {e}")
        individual.fitness = 0.0
        return 0.0


# ============================================================================
# Step 3: Create Initial Population
# ============================================================================

def create_individual() -> Individual:
    """
    Create a random individual

    Ensures fast_period < slow_period to avoid errors

    Returns:
        Randomly generated individual
    """
    while True:
        fast_period = random.randint(FAST_PERIOD_RANGE[0], FAST_PERIOD_RANGE[1])
        slow_period = random.randint(SLOW_PERIOD_RANGE[0], SLOW_PERIOD_RANGE[1])

        # Ensure fast < slow to avoid division by zero
        if fast_period < slow_period:
            break

    leverage = random.uniform(LEVERAGE_RANGE[0], LEVERAGE_RANGE[1])

    return Individual(fast_period, slow_period, leverage)


def create_population(size: int) -> List[Individual]:
    """
    Create initial population

    Args:
        size: Population size

    Returns:
        List of individuals
    """
    population = []
    for _ in range(size):
        individual = create_individual()
        population.append(individual)
    return population


# ============================================================================
# Step 4: Selection
# ============================================================================

def selection(population: List[Individual], elite_size: int) -> List[Individual]:
    """
    Select elite individuals as parents

    Strategy: Keep top N elite individuals

    Args:
        population: Current population
        elite_size: Number of elite individuals to keep

    Returns:
        List of elite individuals
    """
    # Sort by fitness (descending)
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Keep top N
    return population[:elite_size]


# ============================================================================
# Step 5: Crossover
# ============================================================================

def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Create a child from two parents

    Strategy: Randomly inherit genes from each parent

    Args:
        parent1: First parent
        parent2: Second parent

    Returns:
        Child individual
    """
    # Randomly decide which parent to inherit each gene from
    fast_period = random.choice([parent1.fast_period, parent2.fast_period])
    slow_period = random.choice([parent1.slow_period, parent2.slow_period])
    leverage = random.choice([parent1.leverage, parent2.leverage])

    # Ensure fast < slow, swap if needed
    if fast_period >= slow_period:
        fast_period, slow_period = slow_period - 1, slow_period
        if fast_period < FAST_PERIOD_RANGE[0]:
            fast_period = FAST_PERIOD_RANGE[0]
            slow_period = fast_period + 1

    return Individual(fast_period, slow_period, leverage)


# ============================================================================
# Step 6: Mutation
# ============================================================================

def mutate(individual: Individual, mutation_rate: float) -> Individual:
    """
    Mutate an individual

    Strategy: Each gene has mutation_rate probability of random change

    Args:
        individual: Individual to mutate
        mutation_rate: Mutation probability

    Returns:
        Mutated individual
    """
    # Mutate fast period
    if random.random() < mutation_rate:
        individual.fast_period = random.randint(FAST_PERIOD_RANGE[0], FAST_PERIOD_RANGE[1])

    # Mutate slow period
    if random.random() < mutation_rate:
        individual.slow_period = random.randint(SLOW_PERIOD_RANGE[0], SLOW_PERIOD_RANGE[1])

    # Mutate leverage
    if random.random() < mutation_rate:
        individual.leverage = round(random.uniform(LEVERAGE_RANGE[0], LEVERAGE_RANGE[1]), 2)

    # Ensure fast < slow after mutation
    if individual.fast_period >= individual.slow_period:
        # Swap or adjust
        if individual.slow_period > FAST_PERIOD_RANGE[0]:
            individual.fast_period = individual.slow_period - 1
        else:
            individual.slow_period = individual.fast_period + 1

    return individual


# ============================================================================
# Step 7: Main GA Loop
# ============================================================================

def run_ga():
    """Run genetic algorithm optimization"""

    print("\n" + "=" * 80)
    print(" GENETIC ALGORITHM OPTIMIZATION")
    print(" MA CROSSOVER + RSI FILTER STRATEGY")
    print("=" * 80)
    print("\nStrategy: MA Golden/Death Cross with RSI(14) momentum filter")
    print("RSI Filter: Buy when RSI < 70, Sell when RSI > 30")

    # Load training data
    print(f"\nLoading training data: {TRAIN_DATA}")
    df = load_tick_data(TRAIN_DATA, verbose=False)
    df = preprocess_data(df, verbose=False)
    kline_train = resample_to_kline(df, timeframe=TIMEFRAME, verbose=False)
    print(f"Training data: {len(kline_train)} candlesticks")

    # Create initial population
    print(f"\nInitializing population: {POPULATION_SIZE} individuals")
    population = create_population(POPULATION_SIZE)

    # Track best individual in each generation
    best_history = []
    best_fitness_ever = -float('inf')
    stagnation_count = 0
    early_stopped = False

    # Evolution loop
    print(f"\nStarting evolution: {GENERATIONS} generations (early stop patience: {EARLY_STOP_PATIENCE})")
    print("=" * 80)

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}/{GENERATIONS}")

        # Evaluate fitness for all individuals
        for individual in population:
            evaluate_fitness(individual, kline_train)

        # Sort and find best individual
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        best_history.append(best)

        # Early stopping check
        if best.fitness > best_fitness_ever:
            best_fitness_ever = best.fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Print best individual of current generation
        print(f"   Best: MA({best.fast_period}/{best.slow_period}) "
              f"Lev={best.leverage}x -> AR={best.ar:.2f}% MDD={best.mdd:.2f}% "
              f"Sharpe={best.sharpe:.2f} Fitness={best.fitness:.2f} "
              f"(stagnation: {stagnation_count}/{EARLY_STOP_PATIENCE})")

        if stagnation_count >= EARLY_STOP_PATIENCE:
            print(f"\n   Early stopping at generation {generation + 1}: "
                  f"no improvement for {EARLY_STOP_PATIENCE} generations")
            early_stopped = True
            break

        # Skip breeding if last generation
        if generation == GENERATIONS - 1:
            break

        # Select elite individuals
        elites = selection(population, ELITE_SIZE)

        # Generate new population
        new_population = elites.copy()  # Preserve elite

        while len(new_population) < POPULATION_SIZE:
            # Randomly select two parents
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # Crossover to produce child
            child = crossover(parent1, parent2)

            # Mutate
            child = mutate(child, MUTATION_RATE)

            new_population.append(child)

        population = new_population

    # Final results
    print("\n" + "=" * 80)
    if early_stopped:
        print(f" OPTIMIZATION COMPLETED (Early Stopped at gen {len(best_history)})")
    else:
        print(f" OPTIMIZATION COMPLETED ({GENERATIONS} generations)")
    print("=" * 80)

    best_individual = best_history[-1]
    print(f"\nBest Individual:")
    print(f"   MA({best_individual.fast_period}/{best_individual.slow_period})")
    print(f"   Leverage: {best_individual.leverage}x")
    print(f"   Annual Return: {best_individual.ar:.2f}%")
    print(f"   Max Drawdown: {best_individual.mdd:.2f}%")
    print(f"   Sharpe Ratio: {best_individual.sharpe:.2f}")
    print(f"   Sortino Ratio: {best_individual.sortino:.2f}")

    # Validate on test set
    print("\n" + "=" * 80)
    print(" VALIDATION ON TEST DATA (2025)")
    print("=" * 80)

    df_test = load_tick_data(TEST_DATA, verbose=False)
    df_test = preprocess_data(df_test, verbose=False)
    kline_test = resample_to_kline(df_test, timeframe=TIMEFRAME, verbose=False)

    test_fitness = evaluate_fitness(best_individual, kline_test)

    print(f"\nTest Results:")
    print(f"   Annual Return: {best_individual.ar:.2f}%")
    print(f"   Max Drawdown: {best_individual.mdd:.2f}%")
    print(f"   Sharpe Ratio: {best_individual.sharpe:.2f}")
    print(f"   Sortino Ratio: {best_individual.sortino:.2f}")

    # Save results to output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ga_results_{timestamp}.json"

    # Save test metrics BEFORE re-evaluating on training data
    test_metrics = {
        "ar": best_individual.ar,
        "mdd": best_individual.mdd,
        "sharpe": best_individual.sharpe,
        "sortino": best_individual.sortino
    }

    # Re-evaluate on training data to get training metrics
    evaluate_fitness(best_individual, kline_train)
    train_metrics = {
        "ar": best_individual.ar,
        "mdd": best_individual.mdd,
        "sharpe": best_individual.sharpe,
        "sortino": best_individual.sortino
    }

    actual_generations = len(best_history)

    results = {
        "timestamp": timestamp,
        "ga_parameters": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "actual_generations": actual_generations,
            "early_stopped": early_stopped,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "elite_size": ELITE_SIZE,
            "mutation_rate": MUTATION_RATE,
            "max_mdd": MAX_MDD
        },
        "best_parameters": {
            "fast_period": best_individual.fast_period,
            "slow_period": best_individual.slow_period,
            "leverage": best_individual.leverage,
            "timeframe": TIMEFRAME
        },
        "training_performance": train_metrics,
        "test_performance": test_metrics
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print(" DONE!")
    print("=" * 80)

    return best_individual, best_history


# ============================================================================
# Main Program
# ============================================================================

if __name__ == "__main__":
    best_individual, history = run_ga()
