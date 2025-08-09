import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Wrench } from "lucide-react";

const skillsData = [
  {
    title: "Programming Languages",
    skills: ["Python", "R", "SQL", "JavaScript", "TypeScript"],
  },
  {
    title: "Libraries & Frameworks",
    skills: ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "React", "Next.js", "Flask"],
  },
  {
    title: "Databases & Big Data",
    skills: ["PostgreSQL", "MySQL", "MongoDB", "Apache Spark", "Hadoop"],
  },
  {
    title: "Tools & Platforms",
    skills: ["Jupyter Notebook", "Tableau", "Power BI", "Docker", "Git", "AWS", "Google Cloud"],
  },
];

export function TechnicalSkillsSection() {
  return (
    <section id="skills" className="container mx-auto py-16">
      <div className="text-center">
        <h2 className="font-headline text-3xl font-bold">Technical Skills</h2>
        <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
          My toolbox for building data-driven solutions.
        </p>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2">
        {skillsData.map((category) => (
          <Card key={category.title} className="bg-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-headline text-xl">
                <Wrench className="h-5 w-5 text-primary" />
                {category.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {category.skills.map((skill) => (
                  <Badge key={skill} variant="outline" className="border-accent text-accent hover:bg-accent hover:text-accent-foreground">
                    {skill}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
