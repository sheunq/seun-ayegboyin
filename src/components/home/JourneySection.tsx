import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Briefcase, Download, GraduationCap } from "lucide-react";

const workExperience = [
  {
    role: "Data Scientist",
    company: "Tek Tunnel",
    period: "2022 - Present",
    description: [
      "Lead data science projects from conception to deployment.",
      "Conducted Data cleaning, feature engineering, A/B testing\
SQL, NoSQL, Big Data tools (Spark, Hadoop)",
      "Develop and implement machine learning models for predictive analytics.",
      "Model deployment (Flask, FastAPI, Docker)\
Developed CI/CD pipelines, monitoring (MLflow, Kubeflow)",
      "Mentor junior data scientists and contribute to team growth.",
    ],
  },
  {
    role: "Data Scientist Intern",
    company: "British Airways",
    period: "2022 - 2022",
    description: [
      "Developed machine learning models utilizing Logistic Regression and Random Forests to identify customer characteristics/behavior to derive striking insights",
      "Extracted data with BeautifulSoup and Automated data with Selenium for sentiment analysis with TensorFlow.",
      "Built data visualizations and dashboards to communicate findings.",
      "Designed and developed data wrangling and visualization techniques as well as a classification engine based on Logistic Regression",
      "Collaborated with cross-functional teams to solve business problems.",
    ],
  },

  {
    role: "Data Analyst Intern",
    company: "KPMG",
    period: "2021 - 2021",
    description: [
      "Analyzed large datasets to extract actionable insights.",
      "Generated a variety of business reports using SQL queries, Excel, Power BI Desktop(Dashboards), and PowerPoint.",
      "Built data visualizations and dashboards to communicate findings.",
      "Collaborated with cross-functional teams to solve business problems.",
    ],
  },
];

const education = [
    {
    degree: "B.Tech. in Transportation Management Technology",
    institution: "Federal University of Technology Akure",
    period: "2014 - 2019",
  },
];

export function JourneySection() {
  return (
    <section id="journey" className="container mx-auto py-16">
      <div className="text-center">
        <h2 className="font-headline text-3xl font-bold">My Journey</h2>
        <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
          A brief history of my professional and academic background.
        </p>
        <Button asChild className="mt-6" variant="outline">
          <a href="/cv.pdf" download>
            <Download className="mr-2 h-4 w-4" />
            Download Full CV
          </a>
        </Button>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-12 md:grid-cols-2">
        <div>
          <h3 className="flex items-center gap-2 font-headline text-2xl font-semibold">
            <Briefcase className="h-6 w-6 text-primary" />
            Work Experience
          </h3>
          <div className="mt-8 space-y-8">
            {workExperience.map((job) => (
              <Card key={job.role}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <h4 className="font-headline text-xl font-bold">{job.role}</h4>
                    <p className="text-sm text-muted-foreground">{job.period}</p>
                  </div>
                  <p className="text-md text-muted-foreground">{job.company}</p>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc space-y-2 pl-5 text-muted-foreground">
                    {job.description.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <h3 className="flex items-center gap-2 font-headline text-2xl font-semibold">
            <GraduationCap className="h-6 w-6 text-primary" />
            Education
          </h3>
          <div className="mt-8 space-y-8">
            {education.map((edu) => (
              <Card key={edu.degree}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <h4 className="font-headline text-xl font-bold">{edu.degree}</h4>
                    <p className="text-sm text-muted-foreground">{edu.period}</p>
                  </div>
                  <p className="text-md text-muted-foreground">{edu.institution}</p>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
