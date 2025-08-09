import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import Image from "next/image";

export function AboutSection() {
  return (
    <section id="about" className="container mx-auto py-16">
      <div className="grid grid-cols-1 items-center gap-12 md:grid-cols-3">
        <div className="flex justify-center md:col-span-1">
          <Avatar className="h-48 w-48 border-4 border-primary/10">
            <Image 
              src="/images/seun passport.jpg"
              alt="Portrait of the data scientist"
              data-ai-hint="person portrait"
              width={200}
              height={200}
              className="object-cover"
            />
            <AvatarFallback>JD</AvatarFallback>
          </Avatar>
        </div>
        <div className="md:col-span-2">
          <h2 className="font-headline text-3xl font-bold">About Me</h2>
          <p className="mt-4 text-muted-foreground">
            I am a data scientist with a passion for extracting insights from complex datasets. My experience lies in building end-to-end machine learning pipelines, from data wrangling and feature engineering to model deployment and monitoring. I'm driven by curiosity and a desire to solve real-world problems with data-driven solutions.
          </p>
          <Button asChild className="mt-8 bg-accent text-accent-foreground hover:bg-accent/90">
            <a href="/resume.pdf" download>
              <Download className="mr-2 h-4 w-4" />
              Download Resume
            </a>
          </Button>
        </div>
      </div>
    </section>
  );
}
