using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace Infer.NET_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            // 6 maçta kazananlar ve kaybedenler
            var winnerData = new[] { 0, 0, 0, 1, 3, 4 };
            var loserData = new[] { 1, 3, 4, 2, 1, 2 };

            // İstatistiksel modeli, olasılığa dayalı bir program olarak tanımlıyoruz
            var game = new Range(winnerData.Length);
            var player = new Range(winnerData.Concat(loserData).Max() + 1);
            var playerSkills = Variable.Array<double>(player);
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);

            var winners = Variable.Array<int>(game);
            var losers = Variable.Array<int>(game);

            using (Variable.ForEach(game))
            {
                // Takım performanslarının gürültülü(noisy) sonuçlarını alıyoruz
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);

                // Kazananların performansının daha iyi olduğunu öğretiyoruz
                Variable.ConstrainTrue(winnerPerformance > loserPerformance);
            }

            // Verileri modele ekliyoruz
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            // Run inference
            var inferenceEngine = new InferenceEngine();
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            // The inferred skills are uncertain, which is captured in their variance
            var orderedPlayerSkills = inferredSkills
        .Select((s, i) => new { Player = i, Skill = s })
        .OrderByDescending(ps => ps.Skill.GetMean());

            foreach (var playerSkill in orderedPlayerSkills)
            {
                Console.WriteLine($"Takım {playerSkill.Player} skill: {playerSkill.Skill}");
            }
        }
    }
}
