using System;
using System.Collections.Generic;
using System.Globalization;
using OxyBrain;
using System.IO;


namespace TestApp // Note: actual namespace depends on the project name.
{


	class Program
    {
        static void Main(string[] args)
        {
			var network = new NeuronNet(10,new List<int>{10},10);
            var input = new List<double>();
            var output = new List<double>();

            for (int i = 0; i < 10; i++)
            {
                input.Add(3.00);
                output.Add(0);
            }


            for (int i = 0; i < 100; i++)
            {
                            foreach (var item in network.run(input))
            {
                Console.WriteLine("Wynik: " + item);
            }

            network.learn(output);
            }
        }
    }
}