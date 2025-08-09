import SummarizerForm from './SummarizerForm';

export default function SummarizerPage() {
  return (
    <div className="container mx-auto max-w-3xl py-12">
      <div className="text-center">
        <h1 className="font-headline text-4xl font-bold tracking-tight">AI Project Summarizer</h1>
        <p className="mt-4 text-lg text-muted-foreground">
          Tailor your project descriptions for any audience. Paste your detailed project description, select an audience, and let AI craft the perfect summary.
        </p>
      </div>

      <div className="mt-12">
        <SummarizerForm />
      </div>
    </div>
  );
}
