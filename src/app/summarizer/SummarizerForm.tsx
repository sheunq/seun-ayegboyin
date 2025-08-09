'use client';

import { useEffect, useRef, useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { summarizeProjectAction } from '@/app/actions';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { Bot, Loader2, Sparkles } from 'lucide-react';

const initialState = {
  success: false,
};

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full">
      {pending ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Summarizing...
        </>
      ) : (
        <>
          <Sparkles className="mr-2 h-4 w-4" />
          Generate Summary
        </>
      )}
    </Button>
  );
}

export default function SummarizerForm() {
  const [state, formAction] = useActionState(summarizeProjectAction, initialState);
  const { toast } = useToast();
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (!state.success && state.message) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: state.message,
      });
    }
    if (state.success) {
      // Uncomment the line below if you want to clear the form on success
      // formRef.current?.reset();
    }
  }, [state, toast]);

  return (
    <Card>
      <form action={formAction} ref={formRef}>
        <CardHeader>
          <CardTitle>Project Details</CardTitle>
          <CardDescription>Provide the details of your project below.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="projectDescription">Project Description</Label>
            <Textarea
              id="projectDescription"
              name="projectDescription"
              placeholder="e.g., Developed a machine learning model to predict customer churn using Python, Scikit-learn, and Pandas..."
              rows={8}
              required
            />
          </div>
          <div className="space-y-2">
            <Label>Audience Type</Label>
            <RadioGroup name="audienceType" defaultValue="non-technical" className="flex gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="non-technical" id="r1" />
                <Label htmlFor="r1">Non-Technical</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="technical" id="r2" />
                <Label htmlFor="r2">Technical</Label>
              </div>
            </RadioGroup>
          </div>
        </CardContent>
        <CardFooter>
          <SubmitButton />
        </CardFooter>
      </form>
      {state.summary && (
        <div className="border-t p-6">
           <div className="flex items-center gap-2 mb-4">
            <Bot className="h-6 w-6 text-primary" />
            <h3 className="font-headline text-xl font-semibold">Generated Summary</h3>
           </div>
          <div className="prose prose-sm max-w-none rounded-md bg-muted p-4 text-muted-foreground">
            {state.summary}
          </div>
        </div>
      )}
    </Card>
  );
}
