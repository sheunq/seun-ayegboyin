import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export function HeroSection() {
  return (
    <section className="container mx-auto mt-16 text-center sm:mt-24">
      <h1 className="font-headline text-4xl font-bold tracking-tight text-foreground sm:text-5xl md:text-6xl">
        Transforming Data into
        <br />
        <span className="text-primary">Actionable Insights</span>
      </h1>
      <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
        Welcome to my digital space. I'm a data scientist passionate about uncovering stories from data and building intelligent solutions. Explore my projects and articles to see my work in action.
      </p>
      <div className="mt-8 flex justify-center gap-4">
        <Button asChild size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
          <Link href="#contact">
            Get in Touch
          </Link>
        </Button>
        <Button asChild size="lg" variant="outline">
          <Link href="#projects">
            View Projects <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </Button>
      </div>
    </section>
  );
}
