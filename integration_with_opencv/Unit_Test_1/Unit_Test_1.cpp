#include "pch.h"
#include "CppUnitTest.h"
#include "../angle_accuracy/Statistics.h"
#include "../angle_accuracy/Statistics.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest1
{
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			Statistics myStat;

			myStat.add(0);
			myStat.add(1);
			myStat.add(2);
			myStat.add(3);
			myStat.add(4);

			myStat.print_stats(std::to_string(0));

			Assert::IsTrue(myStat.get_average() == 2, L"Average");
			Assert::IsTrue(myStat.get_average_filtered() == 2, L"Average filtered");

			Assert::IsTrue(myStat.get_median() == 2, L"Median");
		}

		TEST_METHOD(TestMethod2)
		{
			Statistics myStat;

			myStat.add(0);
			myStat.add(1);
			myStat.add(2);
			myStat.add(3);

			Assert::IsTrue(myStat.get_median() == 1.5, L"Median uneven");
		}

		TEST_METHOD(TestMethod3)
		{
			Statistics myStat;
			myStat.add(0);
			myStat.add(1);
			myStat.add(2);
			myStat.add(3);
			myStat.add(4);

			std::string s1 = "avg: " + std::to_string( myStat.get_average() ) + "\n";
			Logger::WriteMessage(s1.c_str());

			std::string s = "std dev: " + std::to_string(myStat.get_std_dev());
			Logger::WriteMessage(s.c_str());
			//Assert::IsTrue(myStat.get_median() == 0, L"Median uneven");
		}


		TEST_METHOD(TestMethod4)
		{
			Statistics myStat;
			Assert::IsTrue(myStat.get_median() == 0, L"Median uneven");
		}
	};
}
